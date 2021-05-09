use std::fmt;
use std::ops;

use std::cell::Ref;
use std::cell::RefCell;
use std::cell::RefMut;
use std::rc::Rc;

#[derive(Debug)]
pub struct Variable {
    pub data: f32,
    pub grad: f32,
    pub left_root: Option<VariableRef>,
    pub right_root: Option<VariableRef>,
    pub op: Option<Operator>,
}

// ********************** INIT **********************************
impl Variable {
    fn new_node(
        data: f32,
        left_root: Option<VariableRef>,
        right_root: Option<VariableRef>,
        op: Option<Operator>,
    ) -> Variable {
        Variable {
            data,
            grad: 0.0,
            left_root,
            right_root,
            op,
        }
    }
}

impl Variable {
    // TODO new should be private
    pub fn new(data: f32) -> Variable {
        Variable::new_node(data, None, None, None)
    }
}

#[derive(Debug, Clone)]
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
            self.backward_op(1.0);
        } else {
            self.backward_op(self.grad);
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
    pub fn backward_op(&mut self, grad: f32) {
        match &self.op {
            Some(op) => match op {
                Operator::ADD => {
                    for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
                        match some_var {
                            Some(var) => var.borrow_mut().grad += grad,
                            None => (),
                        }
                    }
                }

                Operator::SUB => {
                    match &mut self.left_root {
                        Some(var) => {
                            var.borrow_mut().grad += grad;
                        }
                        None => (),
                    }

                    match &mut self.right_root {
                        Some(var) => {
                            var.borrow_mut().grad -= grad;
                        }
                        None => (),
                    }
                }
                Operator::MUL => match (&mut self.left_root, &mut self.right_root) {
                    (Some(left), Some(right)) => {
                        let mut right_var = right.borrow_mut();
                        let mut left_var = left.borrow_mut();

                        right_var.grad += left_var.data * grad;
                        left_var.grad += right_var.data * grad;
                    }
                    _ => (),
                },
                Operator::DIV => match (&mut self.left_root, &mut self.right_root) {
                    (Some(left), Some(right)) => {
                        let mut right_var = right.borrow_mut();
                        let mut left_var = left.borrow_mut();

                        if left_var.data == 0.0 {
                            panic!("can't differentiate when divinding by zero");
                        }

                        left_var.grad += grad / right_var.data;

                        right_var.grad -=
                            (grad * left_var.data) / (right_var.data * right_var.data);
                    }
                    _ => (),
                },
            },
            None => (),
        }
    }
}

// *************************** OPERATOR *********************

#[derive(Debug)]
pub enum Operator {
    ADD,
    SUB,
    MUL,
    DIV,
}

impl<'a, 'b> ops::Add<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn add(self, rhs: &'b VariableRef) -> VariableRef {
        VariableRef::new(Variable::new_node(
            self.borrow().data + rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::ADD),
        ))
    }
}

impl<'a, 'b> ops::Sub<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: &'b VariableRef) -> VariableRef {
        VariableRef::new(Variable::new_node(
            self.borrow().data - rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::SUB),
        ))
    }
}

impl<'a, 'b> ops::Mul<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: &'b VariableRef) -> VariableRef {
        VariableRef::new(Variable::new_node(
            self.borrow().data * rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::MUL),
        ))
    }
}

impl<'a, 'b> ops::Div<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: &'b VariableRef) -> VariableRef {
        if rhs.borrow().data == 0.0 {
            panic!("can't divide by zero");
        }

        VariableRef::new(Variable::new_node(
            self.borrow().data / rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::DIV),
        ))
    }
}

impl<'a> ops::Add<f32> for &'a mut VariableRef {
    type Output = VariableRef;

    fn add(self, float_to_add: f32) -> VariableRef {
        self.borrow_mut().data += float_to_add;
        self.clone()
    }
}

impl<'a> ops::Sub<f32> for &'a mut VariableRef {
    type Output = VariableRef;

    fn sub(self, float_to_sub: f32) -> VariableRef {
        self.borrow_mut().data -= float_to_sub;
        self.clone()
    }
}

impl<'a> ops::Mul<f32> for &'a mut VariableRef {
    type Output = VariableRef;

    fn mul(self, float: f32) -> VariableRef {
        self.borrow_mut().data *= float;
        self.clone()
    }
}

impl<'a> ops::Div<f32> for &'a mut VariableRef {
    type Output = VariableRef;

    fn div(self, float: f32) -> VariableRef {
        if float == 0.0 {
            panic!("can't divide by zero");
        }

        self.borrow_mut().data /= float;
        self.clone()
    }
}

// // ************************ unit tests ******************************

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_leaf() {
        let x = Variable::new(2.0);
        assert_eq!(true, x.is_leaf());
    }

    #[test]
    fn new_node_is_not_leaf() {
        let ref x = VariableRef::new(Variable::new(2.0));
        let ref y = VariableRef::new(Variable::new(2.0));

        assert_eq!(false, (x + y).borrow().is_leaf());
    }

    #[test]
    fn add_check_value() {
        let ref x = VariableRef::new(Variable::new(1.0));
        let ref y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data + y.borrow().data;

        assert_eq!(lhs, (x + y).borrow().data);
    }

    #[test]
    fn sub_check_value() {
        let ref x = VariableRef::new(Variable::new(1.0));
        let ref y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data - y.borrow().data;
        assert_eq!(lhs, (x - y).borrow().data);
    }

    #[test]
    fn mul_check_value() {
        let ref x = VariableRef::new(Variable::new(1.0));
        let ref y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data * y.borrow().data;
        assert_eq!(lhs, (x * y).borrow().data);
    }

    #[test]
    fn div_check_value() {
        let ref x = VariableRef::new(Variable::new(1.0));
        let ref y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data / y.borrow().data;
        assert_eq!(lhs, (x / y).borrow().data);
    }

    #[test]
    #[should_panic]
    fn div_check_panic_div_zero() {
        let ref x = VariableRef::new(Variable::new(1.0));
        let ref y = VariableRef::new(Variable::new(0.0));
        let _div = x / y;
    }

    //**********test backward***********

    #[test]
    fn add_check_backward() {
        let ref x = VariableRef::new(Variable::new(2.0));
        let ref y = VariableRef::new(Variable::new(3.0));

        let mut z = x + y;

        z.backward();

        assert_eq!(x.borrow().grad, 1.0);
        assert_eq!(y.borrow().grad, 1.0);
    }

    #[test]
    fn sub_check_backward() {
        let ref x = VariableRef::new(Variable::new(2.0));
        let ref y = VariableRef::new(Variable::new(3.0));

        let mut z = x - y;

        z.backward();

        assert_eq!(x.borrow().grad, 1.0);
        assert_eq!(y.borrow().grad, -1.0);
    }

    #[test]
    fn mul_check_backward() {
        let ref x = VariableRef::new(Variable::new(2.0));
        let ref y = VariableRef::new(Variable::new(3.0));

        let mut z = x * y;

        z.backward();

        assert_eq!(x.borrow().grad, 3.0);
        assert_eq!(y.borrow().grad, 2.0);
    }

    #[test]
    fn div_check_backward() {
        let ref x = VariableRef::new(Variable::new(2.0));
        let ref y = VariableRef::new(Variable::new(3.0));

        let mut z = x / y;

        z.backward();

        assert_eq!(x.borrow().grad, 1.0 / 3.0);
        assert_eq!(y.borrow().grad, -2.0 / 9.0);
    }
}
