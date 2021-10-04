use std::fmt;

use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::rc::Rc;

use ndarray::IxDyn;
use ndarray::{Array, Ix1, NdFloat};

pub struct Variable<T>
where
    T: NdFloat,
{
    pub data: Array<T, IxDyn>,
    pub grad: Option<Array<T, IxDyn>>,
    pub left_root: Option<VariableRef<T>>,
    pub right_root: Option<VariableRef<T>>,
    pub grad_fn: Option<Box<dyn GradFn<T>>>,
}

// ********************** INIT **********************************
impl<T> Variable<T>
where
    T: NdFloat,
{
    fn new_node_i(
        data: Array<T, IxDyn>,
        left_root: Option<VariableRef<T>>,
        right_root: Option<VariableRef<T>>,
        grad_fn: Option<Box<dyn GradFn<T>>>,
        retain_grad: bool,
    ) -> VariableRef<T> {
        let grad = match retain_grad {
            true => Some(Variable::init_grad_value(&data)),
            false => None,
        };

        let var = Variable {
            data,
            grad,
            left_root,
            right_root,
            grad_fn,
        };

        VariableRef::new(var)
    }
    fn new_node(
        data: Array<T, IxDyn>,
        left_root: Option<VariableRef<T>>,
        right_root: Option<VariableRef<T>>,
        grad_fn: Option<Box<dyn GradFn<T>>>,
    ) -> VariableRef<T> {
        Variable::new_node_i(data, left_root, right_root, grad_fn, false)
    }

    pub fn new(data: Array<T, IxDyn>) -> VariableRef<T> {
        Variable::new_node_i(data, None, None, None, true)
    }

    pub fn new_no_retain_grad(data: Array<T, IxDyn>) -> VariableRef<T> {
        Variable::new_node_i(data, None, None, None, false)
    }

    pub fn init_grad_value(data: &Array<T, IxDyn>) -> Array<T, IxDyn> {
        Array::<T, IxDyn>::zeros(data.raw_dim())
    }

    pub fn zero_grad(&mut self) {
        if self.is_grad_retain() {
            self.grad = Some(Variable::init_grad_value(&self.data));
        } else {
            println!(
                "WARNING : zero grad on a Variable which does not retain its grad has no effect"
            );
        }
    }
}

#[derive(Clone)]
pub struct VariableRef<T>
where
    T: NdFloat,
{
    ref_: Rc<RefCell<Variable<T>>>,
}

impl<T> VariableRef<T>
where
    T: NdFloat,
{
    pub fn new(var: Variable<T>) -> VariableRef<T> {
        VariableRef {
            ref_: Rc::new(RefCell::new(var)),
        }
    }

    pub fn borrow(&self) -> Ref<Variable<T>> {
        self.ref_.borrow()
    }

    pub fn borrow_mut(&mut self) -> RefMut<Variable<T>> {
        self.ref_.borrow_mut()
    }

    pub fn backward(&mut self) {
        self.borrow_mut().backward();
    }
}

// *********************** DISPLAY ***********************
impl<T> fmt::Display for Variable<T>
where
    T: NdFloat,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.grad {
            Some(grad) => write!(f, "Variable( {} grad : {})", self.data, grad),
            _ => write!(f, "Variable({})", self.data),
        }
    }
}

impl<T> fmt::Display for VariableRef<T>
where
    T: NdFloat,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

// ****************************MODULE***********************

pub trait GradFn<T>
where
    T: NdFloat,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, IxDyn>, y: &'b Array<T, IxDyn>) -> Array<T, IxDyn>;

    fn backward<'a>(
        &self,
        grad: &'a Array<T, IxDyn>,
        left_ref: &'a VariableRef<T>,
        right_ref: &'a VariableRef<T>,
    ) -> [Array<T, IxDyn>; 2];

    fn subscribe<'a, 'b>(
        &self,
        lhs: &'a VariableRef<T>,
        rhs: &'b VariableRef<T>,
        grad_fn_box: Box<dyn GradFn<T>>,
    ) -> VariableRef<T> {
        Variable::<T>::new_node(
            self.forward(&lhs.borrow().data, &rhs.borrow().data),
            Some(lhs.clone()),
            Some(rhs.clone()),
            Some(grad_fn_box),
        )
    }
}

//**************************** BACKWARD *********************

impl<T> Variable<T>
where
    T: NdFloat,
{
    pub fn is_leaf(&self) -> bool {
        self.right_root.is_none() & self.left_root.is_none()
    }

    pub fn is_grad_retain(&self) -> bool {
        match self.grad {
            Some(_) => true,
            _ => false,
        }
    }

    pub fn retain_grad(&mut self) {
        self.grad = Some(Variable::init_grad_value(&self.data));
    }

    pub fn get_grad(&self) -> Result<Array<T, IxDyn>, &str> {
        match &self.grad {
            Some(grad) => Ok(grad.clone()),
            None => Err("This variable does not requires grad"),
        }
    }

    pub fn get_grad_f(&self) -> Array<T, IxDyn> {
        self.get_grad().unwrap()
    }

    pub fn backward_grad_fn<'a>(&mut self, grad: &Array<T, IxDyn>) -> [Array<T, IxDyn>; 2] {
        let mut grads_to_add: [Array<T, IxDyn>; 2] = [
            Array::<T, Ix1>::zeros(1).into_dyn(),
            Array::<T, Ix1>::zeros(1).into_dyn(),
        ];

        match &self.grad_fn {
            Some(grad_fn) => match (&mut self.left_root, &mut self.right_root) {
                (Some(left_ref), Some(right_ref)) => {
                    grads_to_add = grad_fn.backward(&grad, left_ref, right_ref);

                    // call the borrow_mut in two different scopes so that when left and right left_root target the same variable it does not throw a BorrowMutError
                    {
                        let mut left_var = left_ref.borrow_mut();

                        match &mut left_var.grad {
                            Some(grad) => {
                                *grad += &grads_to_add[0];
                            }
                            _ => (),
                        }
                    }

                    {
                        let mut right_var = right_ref.borrow_mut();

                        match &mut right_var.grad {
                            Some(grad) => {
                                *grad += &grads_to_add[1];
                            }
                            _ => (),
                        }
                    }
                }
                (_, None) => (),
                (None, _) => (),
            },
            None => (),
        }
        grads_to_add
    }

    pub fn backward(&mut self) {
        self.backward_in(&Array::<T, IxDyn>::ones(self.data.raw_dim()));
    }

    fn backward_in(&mut self, grad: &Array<T, IxDyn>) {
        let new_grad = self.backward_grad_fn(grad);

        for (i, some_var) in vec![&mut self.left_root, &mut self.right_root]
            .iter_mut()
            .enumerate()
        {
            match some_var {
                Some(var) => var.borrow_mut().backward_in(&new_grad[i]),
                None => (),
            }
        }
    }
}

// // ************************ unit tests ******************************

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn new_is_leaf() {
        let x = Variable::new(array!([1.0]).into_dyn());
        assert_eq!(true, x.borrow().is_leaf());
        assert_eq!(true, x.borrow().is_grad_retain());
    }

    #[test]
    fn new_node_is_not_leaf() {
        let ref x = Variable::new(array!([2.0]).into_dyn());
        let ref y = Variable::new(array!([2.0]).into_dyn());

        let z = x + y;
        assert_eq!(false, z.borrow().is_leaf());
        assert_eq!(false, z.borrow().is_grad_retain());
    }

    #[test]
    fn zero_grad() {
        let ref mut x = Variable::new(array!([2.0]).into_dyn());
        let ref mut y = Variable::new(array!([2.0]).into_dyn());

        let mut z = &x.clone() + y;
        z.backward();
        x.borrow_mut().zero_grad();

        assert_eq!(x.borrow().get_grad().unwrap(), array!([0.0]).into_dyn());
    }
}
