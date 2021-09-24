use std::ops;

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Sub {}

impl Module for Sub {
    fn forward(&self, x: f32, y: f32) -> f32 {
        x - y
    }

    fn backward<'a>(
        &self,
        grad: &'a f32,
        _left_ref: &'a VariableRef,
        _right_ref: &'a VariableRef,
    ) -> [f32; 2] {
        [*grad, -*grad]
    }
}

impl<'a, 'b> ops::Sub<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: &'b VariableRef) -> VariableRef {
        let module = Sub {};
        module.subscribe(self, rhs, Box::new(Sub {}))
    }
}

impl<'a> ops::Sub<&'a VariableRef> for VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: &'a VariableRef) -> VariableRef {
        let module = Sub {};
        module.subscribe(&self, rhs, Box::new(Sub {}))
    }
}

impl<'a> ops::Sub<VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: VariableRef) -> VariableRef {
        let module = Sub {};
        module.subscribe(self, &rhs, Box::new(Sub {}))
    }
}

impl ops::Sub<VariableRef> for VariableRef {
    type Output = VariableRef;

    fn sub(self, rhs: VariableRef) -> VariableRef {
        let module = Sub {};
        module.subscribe(&self, &rhs, Box::new(Sub {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;

    #[test]
    fn sub_check_backward() {
        let ref x = Variable::new(2.0);
        let ref y = Variable::new(3.0);

        let mut z = x - y;

        z.backward();

        assert_eq!(x.borrow().grad, 1.0);
        assert_eq!(y.borrow().grad, -1.0);
    }
}
