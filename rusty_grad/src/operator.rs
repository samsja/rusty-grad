use std::ops;

use crate::variable::Operator;
use crate::variable::Variable;
use crate::variable::VariableRef;

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

// ****** float OP variableRef

impl ops::Add<f32> for VariableRef {
    type Output = VariableRef;

    fn add(mut self, float_to_add: f32) -> VariableRef {
        self.borrow_mut().data += float_to_add;
        self.clone()
    }
}

impl ops::Sub<f32> for VariableRef {
    type Output = VariableRef;

    fn sub(mut self, float_to_sub: f32) -> VariableRef {
        self.borrow_mut().data -= float_to_sub;
        self.clone()
    }
}

impl ops::Mul<f32> for VariableRef {
    type Output = VariableRef;

    fn mul(mut self, float: f32) -> VariableRef {
        self.borrow_mut().data *= float;
        self.clone()
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

    #[test]
    fn add_check_float() {
        let ref mut x = VariableRef::new(Variable::new(1.0));
        let _z = x + 4.0;
    }
}
