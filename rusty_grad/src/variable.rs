use std::fmt;

use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::rc::Rc;

use ndarray::{Array, Dimension, Ix1, NdFloat};

pub struct Variable<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub data: Array<T, D>,
    pub grad: Array<T, D>,
    pub left_root: Option<VariableRef<T, D>>,
    pub right_root: Option<VariableRef<T, D>>,
    pub module: Option<Box<dyn Module<T, D>>>,
}

// ********************** INIT **********************************
impl<T, D> Variable<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn new_node(
        data: Array<T, D>,
        left_root: Option<VariableRef<T, D>>,
        right_root: Option<VariableRef<T, D>>,
        module: Option<Box<dyn Module<T, D>>>,
    ) -> VariableRef<T, D> {
        let grad_zero = data.clone(); //should be zero
        let var = Variable {
            data,
            grad: grad_zero, // should be some zeros function
            left_root,
            right_root,
            module,
        };

        VariableRef::new(var)
    }
}

impl<T, D> Variable<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn new(data: Array<T, D>) -> VariableRef<T, D> {
        Variable::new_node(data, None, None, None)
    }
}

#[derive(Clone)]
pub struct VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    ref_: Rc<RefCell<Variable<T, D>>>,
}

impl<T, D> VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn new(var: Variable<T, D>) -> VariableRef<T, D> {
        VariableRef {
            ref_: Rc::new(RefCell::new(var)),
        }
    }

    pub fn borrow(&self) -> Ref<Variable<T, D>> {
        self.ref_.borrow()
    }

    pub fn borrow_mut(&mut self) -> RefMut<Variable<T, D>> {
        self.ref_.borrow_mut()
    }

    pub fn backward(&mut self) {
        self.borrow_mut().backward();
    }
}

// *********************** DISPLAY ***********************
impl<T, D> fmt::Display for Variable<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Variable( {} grad : {})", self.data, self.grad)
    }
}

impl<T, D> fmt::Display for VariableRef<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

// ****************************MODULE***********************

pub trait Module<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    fn forward<'a, 'b>(&self, x: &'a Array<T, D>, y: &'b Array<T, D>) -> Array<T, D>;

    fn backward<'a>(
        &self,
        grad: &'a Array<T, D>,
        left_ref: &'a VariableRef<T, D>,
        right_ref: &'a VariableRef<T, D>,
    ) -> [f32; 2];

    fn subscribe<'a, 'b>(
        &self,
        lhs: &'a VariableRef<T, D>,
        rhs: &'b VariableRef<T, D>,
        module_box: Box<dyn Module<T, D>>,
    ) -> VariableRef<T, D> {
        Variable::<T, D>::new_node(
            self.forward(&lhs.borrow().data, &rhs.borrow().data),
            Some(lhs.clone()),
            Some(rhs.clone()),
            Some(module_box),
        )
    }
}

//**************************** BACKWARD *********************

impl<T, D> Variable<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn is_leaf(&self) -> bool {
        self.right_root.is_none() & self.left_root.is_none()
    }
}

impl<T, D> Variable<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn backward_module<'a>(&mut self, grad: &'a Array<T, D>) {
        match &self.module {
            Some(module) => match (&mut self.left_root, &mut self.right_root) {
                (Some(left_ref), Some(right_ref)) => {
                    let grads_to_add = module.backward(&grad, left_ref, right_ref);
                    // left_ref.borrow_mut().grad += grads_to_add[0];
                    // right_ref.borrow_mut().grad += grads_to_add[1]
                }
                (_, None) => (),
                (None, _) => (),
            },
            None => (),
        }
    }
}

impl<T, D> Variable<T, D>
where
    T: NdFloat,
    D: Dimension,
{
    pub fn backward(&mut self) {
        self.backward_in(true);
    }

    fn backward_in(&mut self, root: bool) {
        /*         if root { */
        //     self.backward_module(&mut self.grad); //should be ones
        //
        // } else {
        //     self.backward_module(&mut self.grad);
        // }
        /*  */
        let ref grad = self.grad.clone();
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

    #[test]
    fn new_is_leaf() {
        let x = Variable::new(Array::<f32, Ix1>::zeros(5));
        assert_eq!(true, x.borrow().is_leaf());
    }

    #[test]
    fn play() {
        let x = Variable::new(Array::<f32, Ix1>::zeros(5));

        let y = Variable::new(Array::<f32, Ix1>::ones(5));
    }

    // #[test]
    // fn new_node_is_not_leaf() {
    // let ref x = Variable::new(2.0);
    // let ref y = Variable::new(2.0);
    //
    // assert_eq!(false, (x + y).borrow().is_leaf());
    // }
}
