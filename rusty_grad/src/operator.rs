use std::ops;

use crate::variable::Operator;
use crate::variable::Variable;
use crate::variable::VariableRef;

//************* & and &

impl<'a, 'b> ops::Sub<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: &'b VariableRef) -> VariableRef {
        Variable::new_node(
            self.borrow().data - rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::SUB),
        )
    }
}

impl<'a, 'b> ops::Mul<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: &'b VariableRef) -> VariableRef {
        Variable::new_node(
            self.borrow().data * rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::MUL),
        )
    }
}

impl<'a, 'b> ops::Div<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: &'b VariableRef) -> VariableRef {
        if rhs.borrow().data == 0.0 {
            panic!("can't divide by zero");
        }

        Variable::new_node(
            self.borrow().data / rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::DIV),
        )
    }
}

// & and Var

impl<'b> ops::Sub<&'b VariableRef> for VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: &'b VariableRef) -> VariableRef {
        Variable::new_node(
            self.borrow().data - rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::SUB),
        )
    }
}

impl<'b> ops::Mul<&'b VariableRef> for VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: &'b VariableRef) -> VariableRef {
        Variable::new_node(
            self.borrow().data * rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::MUL),
        )
    }
}

impl<'b> ops::Div<&'b VariableRef> for VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: &'b VariableRef) -> VariableRef {
        if rhs.borrow().data == 0.0 {
            panic!("can't divide by zero");
        }

        Variable::new_node(
            self.borrow().data / rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::DIV),
        )
    }
}

// var and &
impl<'b> ops::Sub<VariableRef> for &'b VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: VariableRef) -> VariableRef {
        Variable::new_node(
            self.borrow().data - rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::SUB),
        )
    }
}

impl<'b> ops::Mul<VariableRef> for &'b VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: VariableRef) -> VariableRef {
        Variable::new_node(
            self.borrow().data * rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::MUL),
        )
    }
}

impl<'b> ops::Div<VariableRef> for &'b VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: VariableRef) -> VariableRef {
        if rhs.borrow().data == 0.0 {
            panic!("can't divide by zero");
        }

        Variable::new_node(
            self.borrow().data / rhs.borrow().data,
            Some(self.clone()),
            Some(rhs.clone()),
            Some(Operator::DIV),
        )
    }
}
// // ************************ unit tests ******************************

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_check_value() {
        let ref x = Variable::new(1.0);
        let ref y = Variable::new(2.0);

        let lhs = x.borrow().data - y.borrow().data;
        assert_eq!(lhs, (x - y).borrow().data);
    }

    #[test]
    fn mul_check_value() {
        let ref x = Variable::new(1.0);
        let ref y = Variable::new(2.0);

        let lhs = x.borrow().data * y.borrow().data;
        assert_eq!(lhs, (x * y).borrow().data);
    }

    #[test]
    fn div_check_value() {
        let ref x = Variable::new(1.0);
        let ref y = Variable::new(2.0);

        let lhs = x.borrow().data / y.borrow().data;
        assert_eq!(lhs, (x / y).borrow().data);
    }

    #[test]
    #[should_panic]
    fn div_check_panic_div_zero() {
        let ref x = Variable::new(1.0);
        let ref y = Variable::new(0.0);
        let _div = x / y;
    }
}
