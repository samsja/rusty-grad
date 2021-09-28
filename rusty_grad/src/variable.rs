use std::fmt;

use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::rc::Rc;

use ndarray::{Array, Dimension, NdFloat};

pub struct Variable<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
{
    pub data: Array<T, D>,
    pub grad: Option<Array<T, D>>,
    pub left_root: Option<VariableRef<T, D, Dl, Dr>>,
    pub right_root: Option<VariableRef<T, D, Dl, Dr>>,
    pub module: Option<Box<dyn Module<T, D, Dl, Dr>>>,
}

// ********************** INIT **********************************
impl<T, D, Dl, Dr> Variable<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
{
    fn new_node_i(
        data: Array<T, D>,
        left_root: Option<VariableRef<T, D, Dl, Dr>>,
        right_root: Option<VariableRef<T, D, Dl, Dr>>,
        module: Option<Box<dyn Module<T, D, Dl, Dr>>>,
        requires_grad: bool,
    ) -> VariableRef<T, D, Dl, Dr> {
        let grad = match requires_grad {
            true => Some(Array::<T, D>::zeros(data.raw_dim())),
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
        data: Array<T, D>,
        left_root: Option<VariableRef<T, D, Dl, Dr>>,
        right_root: Option<VariableRef<T, D, Dl, Dr>>,
        module: Option<Box<dyn Module<T, D, Dl, Dr>>>,
    ) -> VariableRef<T, D, Dl, Dr> {
        Variable::new_node_i(data, left_root, right_root, module, true)
    }
}

impl<T, D> Variable<T, D, D, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn new(data: Array<T, D>) -> VariableRef<T, D, D, D> {
        Variable::new_node_i(data, None, None, None, true)
    }

    pub fn new_no_grad(data: Array<T, D>) -> VariableRef<T, D, D, D> {
        Variable::new_node_i(data, None, None, None, false)
    }
}

#[derive(Clone)]
pub struct VariableRef<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
{
    ref_: Rc<RefCell<Variable<T, D, Dl, Dr>>>,
}

impl<T, D, Dl, Dr> VariableRef<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
{
    pub fn new(var: Variable<T, D, Dl, Dr>) -> VariableRef<T, D, Dl, Dr> {
        VariableRef {
            ref_: Rc::new(RefCell::new(var)),
        }
    }

    pub fn borrow(&self) -> Ref<Variable<T, D, Dl, Dr>> {
        self.ref_.borrow()
    }

    pub fn borrow_mut(&mut self) -> RefMut<Variable<T, D, Dl, Dr>> {
        self.ref_.borrow_mut()
    }

    pub fn backward(&mut self) {
        self.borrow_mut().backward();
    }
}

// *********************** DISPLAY ***********************
impl<T, D, Dl, Dr> fmt::Display for Variable<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.grad {
            Some(grad) => write!(f, "Variable( {} grad : {})", self.data, grad),
            _ => write!(f, "Variable( {} , no grad required)", self.data),
        }
    }
}

impl<T, D, Dl, Dr> fmt::Display for VariableRef<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

// ****************************MODULE***********************

pub trait Module<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, D>, y: &'b Array<T, D>) -> Array<T, D>;

    fn backward<'a>(
        &self,
        grad: &'a Array<T, D>,
        left_ref: &'a VariableRef<T, D, Dl, Dr>,
        right_ref: &'a VariableRef<T, D, Dl, Dr>,
    ) -> [Array<T, D>; 2];

    fn subscribe<'a, 'b>(
        &self,
        lhs: &'a VariableRef<T, D, Dl, Dr>,
        rhs: &'b VariableRef<T, D, Dl, Dr>,
        module_box: Box<dyn Module<T, D, Dl, Dr>>,
    ) -> VariableRef<T, D, Dl, Dr> {
        Variable::<T, D, Dl, Dr>::new_node(
            self.forward(&lhs.borrow().data, &rhs.borrow().data),
            Some(lhs.clone()),
            Some(rhs.clone()),
            Some(module_box),
        )
    }
}

//**************************** BACKWARD *********************

impl<T, D, Dl, Dr> Variable<T, D, Dl, Dr>
where
    T: NdFloat,
    D: Dimension,
    Dl: Dimension,
    Dr: Dimension,
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

    pub fn get_grad(&self) -> Result<Array<T, D>, &str> {
        match &self.grad {
            Some(grad) => Ok(grad.clone()),
            None => Err("This variable does not requires grad"),
        }
    }

    pub fn get_grad_f(&self) -> Array<T, D> {
        self.get_grad().unwrap()
    }

    pub fn backward_module<'a>(&mut self, grad: Array<T, D>) {
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
        let grad: Array<T, D>;
        match root {
            true => {
                grad = Array::<T, D>::ones(self.data.raw_dim());
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
        let x = Variable::new(array!([1.0]));
        assert_eq!(true, x.borrow().is_leaf());
    }

    #[test]
    fn new_node_is_not_leaf() {
        let ref x = Variable::new(array!([2.0]));
        let ref y = Variable::new(array!([2.0]));

        assert_eq!(false, (x + y).borrow().is_leaf());
    }
}
