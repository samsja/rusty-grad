use std::ops;

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Mul {}

impl Module for Mul {
    fn forward(&self, x: f32, y: f32) -> f32 {
        x * y
    }

    fn backward<'a>(
        &self,
        grad: &'a f32,
        left_ref: &'a VariableRef,
        right_ref: &'a VariableRef,
    ) -> [f32; 2] {
        let left_var = left_ref.borrow();
        let right_var = right_ref.borrow();

        [right_var.data * (*grad), left_var.data * (*grad)]
    }
}

impl<'a, 'b> ops::Mul<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: &'b VariableRef) -> VariableRef {
        let module = Mul {};
        module.subscribe(self, rhs, Box::new(Mul {}))
    }
}

impl<'a> ops::Mul<&'a VariableRef> for VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: &'a VariableRef) -> VariableRef {
        let module = Mul {};
        module.subscribe(&self, rhs, Box::new(Mul {}))
    }
}

impl<'a> ops::Mul<VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: VariableRef) -> VariableRef {
        let module = Mul {};
        module.subscribe(self, &rhs, Box::new(Mul {}))
    }
}

impl ops::Mul<VariableRef> for VariableRef {
    type Output = VariableRef;

    fn mul(self, rhs: VariableRef) -> VariableRef {
        let module = Mul {};
        module.subscribe(&self, &rhs, Box::new(Mul {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;

    #[test]
    fn mul_check_backward() {
        let ref x = Variable::new(2.0);
        let ref y = Variable::new(3.0);

        let mut z = x * y;

        z.backward();

        assert_eq!(x.borrow().grad, 3.0);
        assert_eq!(y.borrow().grad, 2.0);
    }
}
