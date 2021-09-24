use std::fmt;

use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::rc::Rc;

pub struct Variable {
    pub data: f32,
    pub grad: f32,
    pub left_root: Option<VariableRef>,
    pub right_root: Option<VariableRef>,
    pub module: Option<Box<dyn Module>>,
}

// ********************** INIT **********************************
impl Variable {
    pub fn new_node(
        data: f32,
        left_root: Option<VariableRef>,
        right_root: Option<VariableRef>,
        module: Option<Box<dyn Module>>,
    ) -> VariableRef {
        let var = Variable {
            data,
            grad: 0.0,
            left_root,
            right_root,
            module,
        };

        VariableRef::new(var)
    }
}

impl Variable {
    pub fn new(data: f32) -> VariableRef {
        Variable::new_node(data, None, None, None)
    }
}

#[derive(Clone)]
pub struct VariableRef {
    ref_: Rc<RefCell<Variable>>,
}

impl VariableRef {
    pub fn new(var: Variable) -> VariableRef {
        VariableRef {
            ref_: Rc::new(RefCell::new(var)),
        }
    }

    pub fn borrow(&self) -> Ref<Variable> {
        self.ref_.borrow()
    }

    pub fn borrow_mut(&mut self) -> RefMut<Variable> {
        self.ref_.borrow_mut()
    }

    pub fn backward(&mut self) {
        self.borrow_mut().backward();
    }
}

// *********************** DISPLAY ***********************
impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Variable( {} grad : {})", self.data, self.grad)
    }
}

impl fmt::Display for VariableRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

// ****************************MODULE***********************

pub trait Module {
    fn forward(&self, x: f32, y: f32) -> f32;

    fn backward<'a>(
        &self,
        grad: &'a f32,
        left_ref: &'a VariableRef,
        right_ref: &'a VariableRef,
    ) -> [f32; 2];

    fn subscribe<'a, 'b>(
        &self,
        lhs: &'a VariableRef,
        rhs: &'b VariableRef,
        module_box: Box<dyn Module>,
    ) -> VariableRef {
        Variable::new_node(
            self.forward(lhs.borrow().data, rhs.borrow().data),
            Some(lhs.clone()),
            Some(rhs.clone()),
            Some(module_box),
        )
    }
}

//**************************** BACKWARD *********************

impl Variable {
    pub fn is_leaf(&self) -> bool {
        self.right_root.is_none() & self.left_root.is_none()
    }
}

impl Variable {
    /// this is the public backward it is equivalent to the private backward_in(1.0)
    pub fn backward(&mut self) {
        self.backward_in(true);
    }

    fn backward_in(&mut self, root: bool) {
        if root {
            self.backward_module(1.0);
        } else {
            self.backward_module(self.grad);
        }

        for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
            match some_var {
                Some(var) => var.borrow_mut().backward_in(false),
                None => (),
            }
        }
    }
}

impl Variable {
    pub fn backward_module(&mut self, grad: f32) {
        match &self.module {
            Some(module) => match (&mut self.left_root, &mut self.right_root) {
                (Some(left_ref), Some(right_ref)) => {
                    let grads_to_add = module.backward(&grad, left_ref, right_ref);
                    left_ref.borrow_mut().grad += grads_to_add[0];
                    right_ref.borrow_mut().grad += grads_to_add[1]
                }
                (_, None) => (),
                (None, _) => (),
            },
            None => (),
        }
    }
}
// // ************************ unit tests ******************************

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_leaf() {
        let x = Variable::new(2.0);
        assert_eq!(true, x.borrow().is_leaf());
    }

    // #[test]
    // fn new_node_is_not_leaf() {
    // let ref x = Variable::new(2.0);
    // let ref y = Variable::new(2.0);
    //
    // assert_eq!(false, (x + y).borrow().is_leaf());
    // }

    // #[test]
    // fn div_check_backward() {
    //     let ref x = Variable::new(2.0);
    //     let ref y = Variable::new(3.0);
    //
    //     let mut z = x / y;
    //
    //     z.backward();
    //
    //     assert_eq!(x.borrow().grad, 1.0 / 3.0);
    //     assert_eq!(y.borrow().grad, -2.0 / 9.0);
    /* } */
}
