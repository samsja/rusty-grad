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
    pub fn backward(&mut self) {
        self.backward_op();

        for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
            match some_var {
                Some(var) => var.borrow_mut().backward(),
                None => (),
            }
        }
    }
}

impl Variable {
    pub fn backward_op(&mut self) {
        match &self.op {
            Some(op) => match op {
                Operator::ADD => {
                    for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
                        match some_var {
                            Some(var) => {
                                let data = var.borrow().data;
                                var.borrow_mut().grad += data
                            }
                            None => (),
                        }
                    }
                }

                Operator::SUB => {
                    for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
                        match some_var {
                            Some(var) => {
                                let data = var.borrow().data;
                                var.borrow_mut().grad -= data;
                            }
                            None => (),
                        }
                    }
                }
                Operator::MUL => match (&mut self.left_root, &mut self.right_root) {
                    (Some(left), Some(right)) => {
                        let mut right_var = right.borrow_mut();
                        let mut left_var = left.borrow_mut();

                        right_var.grad += right_var.data;
                        left_var.grad += left_var.data;
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

                        right_var.grad += 1.0 / right_var.data;
                        left_var.grad -= right_var.data / (left_var.data * left_var.data);
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

impl ops::Add<VariableRef> for VariableRef {
    type Output = Variable;

    fn add(self, rhs: VariableRef) -> Variable {
        Variable::new_node(
            self.borrow().data + rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::ADD),
        )
    }
}

impl ops::Sub<VariableRef> for VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: VariableRef) -> VariableRef {
        VariableRef::new(Variable::new_node(
            self.borrow().data - rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::ADD),
        ))
    }
}

impl ops::Mul<VariableRef> for VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: VariableRef) -> VariableRef {
        VariableRef::new(Variable::new_node(
            self.borrow().data * rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::ADD),
        ))
    }
}

impl ops::Div<VariableRef> for VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: VariableRef) -> VariableRef {
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
        let x = VariableRef::new(Variable::new(2.0));
        let y = VariableRef::new(Variable::new(2.0));

        assert_eq!(false, (x + y).is_leaf());
    }

    #[test]
    fn add_check_value() {
        let x = VariableRef::new(Variable::new(1.0));
        let y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data + y.borrow().data;
        assert_eq!(lhs, (x + y).data);
    }

    #[test]
    fn sub_check_value() {
        let x = VariableRef::new(Variable::new(1.0));
        let y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data - y.borrow().data;
        assert_eq!(lhs, (x - y).borrow().data);
    }

    #[test]
    fn mul_check_value() {
        let x = VariableRef::new(Variable::new(1.0));
        let y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data * y.borrow().data;
        assert_eq!(lhs, (x * y).borrow().data);
    }

    #[test]
    fn div_check_value() {
        let x = VariableRef::new(Variable::new(1.0));
        let y = VariableRef::new(Variable::new(2.0));

        let lhs = x.borrow().data / y.borrow().data;
        assert_eq!(lhs, (x / y).borrow().data);
    }

    #[test]
    #[should_panic]
    fn div_check_panic_div_zero() {
        let x = VariableRef::new(Variable::new(1.0));
        let y = VariableRef::new(Variable::new(0.0));
        let _div = x / y;
    }
}
