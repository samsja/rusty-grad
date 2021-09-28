use std::fmt;

use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::rc::Rc;

use ndarray::IxDyn;
use ndarray::{Array, NdFloat};

pub struct Variable<T>
where
    T: NdFloat,
{
    pub data: Array<T, IxDyn>,
    pub grad: Option<Array<T, IxDyn>>,
    pub left_root: Option<VariableRef<T>>,
    pub right_root: Option<VariableRef<T>>,
    pub module: Option<Box<dyn Module<T>>>,
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
        module: Option<Box<dyn Module<T>>>,
        requires_grad: bool,
    ) -> VariableRef<T> {
        let grad = match requires_grad {
            true => Some(Array::<T, IxDyn>::zeros(data.raw_dim())),
            false => None,
        };

        let var = Variable {
            data,
            grad,
            left_root,
            right_root,
            module,
        };

        VariableRef::new(var)
    }
    fn new_node(
        data: Array<T, IxDyn>,
        left_root: Option<VariableRef<T>>,
        right_root: Option<VariableRef<T>>,
        module: Option<Box<dyn Module<T>>>,
    ) -> VariableRef<T> {
        Variable::new_node_i(data, left_root, right_root, module, true)
    }

    pub fn new(data: Array<T, IxDyn>) -> VariableRef<T> {
        Variable::new_node_i(data, None, None, None, true)
    }

    pub fn new_no_grad(data: Array<T, IxDyn>) -> VariableRef<T> {
        Variable::new_node_i(data, None, None, None, false)
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
            _ => write!(f, "Variable( {} , no grad required)", self.data),
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

pub trait Module<T>
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
        module_box: Box<dyn Module<T>>,
    ) -> VariableRef<T> {
        Variable::<T>::new_node(
            self.forward(&lhs.borrow().data, &rhs.borrow().data),
            Some(lhs.clone()),
            Some(rhs.clone()),
            Some(module_box),
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

    pub fn requires_grad(&self) -> bool {
        match self.grad {
            Some(_) => true,
            _ => false,
        }
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

    pub fn backward_module<'a>(&mut self, grad: Array<T, IxDyn>) {
        match &self.module {
            Some(module) => match (&mut self.left_root, &mut self.right_root) {
                (Some(left_ref), Some(right_ref)) => {
                    let grads_to_add = module.backward(&grad, left_ref, right_ref);

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
    }

    pub fn backward(&mut self) {
        self.backward_in(true);
    }

    fn backward_in(&mut self, root: bool) {
        let grad: Array<T, IxDyn>;
        match root {
            true => {
                grad = Array::<T, IxDyn>::ones(self.data.raw_dim());
            }
            false => {
                grad = self.get_grad().unwrap();
            }
        }
        self.backward_module(grad);

        for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
            match some_var {
                Some(var) => var.borrow_mut().backward_in(false),
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
    }

    #[test]
    fn new_node_is_not_leaf() {
        let ref x = Variable::new(array!([2.0]).into_dyn());
        let ref y = Variable::new(array!([2.0]).into_dyn());

        assert_eq!(false, (x + y).borrow().is_leaf());
    }
}
