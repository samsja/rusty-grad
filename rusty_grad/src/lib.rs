use std::fmt;
use std::ops;

#[derive(Debug)]
pub struct Variable<'a> {
    pub data: f32,
    pub grad: f32,
    pub left_root: Option<&'a mut Variable<'a>>,
    pub right_root: Option<&'a mut Variable<'a>>,
    pub op: Option<Operator>,
}

// *********************** DISPLAY ***********************
impl<'a> fmt::Display for Variable<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Variable( {} grad : {})", self.data, self.grad)
    }
}

// ********************** INIT **********************************
impl<'a> Variable<'a> {
    fn new_node(
        data: f32,
        left_root: Option<&'a mut Variable<'a>>,
        right_root: Option<&'a mut Variable<'a>>,
        op: Option<Operator>,
    ) -> Variable<'a> {
        Variable {
            data,
            grad: 0.0,
            left_root,
            right_root,
            op,
        }
    }
}

impl<'a> Variable<'a> {
    // TODO new should be private
    pub fn new(data: f32) -> Variable<'a> {
        Variable::new_node(data, None, None, None)
    }
}

//**************************** BACKWARD *********************

impl<'a> Variable<'a> {
    pub fn is_leaf(&self) -> bool {
        self.right_root.is_none() & self.left_root.is_none()
    }
}

impl<'a> Variable<'a> {
    pub fn backward(&mut self) {
        self.backward_op();

        for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
            match some_var {
                Some(var) => var.backward(),
                None => (),
            }
        }
    }
}

impl<'a> Variable<'a> {
    pub fn backward_op(&mut self) {
        match &self.op {
            Some(op) => match op {
                Operator::ADD => {
                    for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
                        match some_var {
                            Some(var) => var.grad += var.data,
                            None => (),
                        }
                    }
                }
                Operator::SUB => {
                    for some_var in vec![&mut self.left_root, &mut self.right_root].iter_mut() {
                        match some_var {
                            Some(var) => var.grad -= var.data,
                            None => (),
                        }
                    }
                }
                Operator::MUL => match (&mut self.left_root, &mut self.right_root) {
                    (Some(left), Some(right)) => {
                        right.grad += left.data;
                        left.grad += right.data;
                    }
                    _ => (),
                },
                Operator::DIV => {
                    unimplemented!();
                }
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

impl<'a> ops::Add<&'a mut Variable<'a>> for &'a mut Variable<'a> {
    type Output = Variable<'a>;

    fn add(self, rhs: &'a mut Variable<'a>) -> Variable<'a> {
        Variable::new_node(
            self.data + rhs.data,
            Some(self),
            Some(rhs),
            Some(Operator::ADD),
        )
    }
}

impl<'a> ops::Sub<&'a mut Variable<'a>> for &'a mut Variable<'a> {
    type Output = Variable<'a>;

    fn sub(self, rhs: &'a mut Variable<'a>) -> Variable<'a> {
        Variable::new_node(
            self.data - rhs.data,
            Some(self),
            Some(rhs),
            Some(Operator::SUB),
        )
    }
}

impl<'a> ops::Mul<&'a mut Variable<'a>> for &'a mut Variable<'a> {
    type Output = Variable<'a>;

    fn mul(self, rhs: &'a mut Variable<'a>) -> Variable<'a> {
        Variable::new_node(
            self.data * rhs.data,
            Some(self),
            Some(rhs),
            Some(Operator::MUL),
        )
    }
}

impl<'a> ops::Div<&'a mut Variable<'a>> for &'a mut Variable<'a> {
    type Output = Variable<'a>;

    fn div(self, rhs: &'a mut Variable<'a>) -> Variable<'a> {
        if rhs.data == 0.0 {
            panic!("can't divide by zero");
        }

        Variable::new_node(
            self.data / rhs.data,
            Some(self),
            Some(rhs),
            Some(Operator::DIV),
        )
    }
}

// ************************ unit tests ******************************

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
        let mut x = Variable::new(2.0);
        let mut y = Variable::new(2.0);

        assert_eq!(false, (&mut x + &mut y).is_leaf());
    }

    #[test]
    fn add_check_value() {
        let mut x = Variable::new(2.0);
        let mut y = Variable::new(2.0);

        assert_eq!(x.data + y.data, (&mut x + &mut y).data);
    }

    #[test]
    fn sub_check_value() {
        let mut x = Variable::new(2.0);
        let mut y = Variable::new(2.0);

        assert_eq!(x.data - y.data, (&mut x - &mut y).data);
    }

    #[test]
    fn mul_check_value() {
        let mut x = Variable::new(2.0);
        let mut y = Variable::new(2.0);

        assert_eq!(x.data * y.data, (&mut x * &mut y).data);
    }

    #[test]
    fn div_check_value() {
        let mut x = Variable::new(2.0);
        let mut y = Variable::new(2.0);

        assert_eq!(x.data / y.data, (&mut x / &mut y).data);
    }

    #[test]
    #[should_panic]
    fn div_check_panic_div_zero() {
        let mut x = Variable::new(2.0);
        let mut y = Variable::new(0.0);
        let _div = &mut x / &mut y;
    }
}
