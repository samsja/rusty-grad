use std::ops;

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Div {}

impl Module for Div {
    fn forward(&self, x: f32, y: f32) -> f32 {
        x / y
    }

    fn backward<'a>(
        &self,
        grad: &'a f32,
        left_ref: &'a VariableRef,
        right_ref: &'a VariableRef,
    ) -> [f32; 2] {
        let left_var = left_ref.borrow();
        let right_var = right_ref.borrow();

        [
            *grad / right_var.data,
            -((*grad) * left_var.data) / (right_var.data * right_var.data),
        ]
    }
}

impl<'a, 'b> ops::Div<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: &'b VariableRef) -> VariableRef {
        let module = Div {};
        module.subscribe(self, rhs, Box::new(Div {}))
    }
}

impl<'a> ops::Div<&'a VariableRef> for VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: &'a VariableRef) -> VariableRef {
        let module = Div {};
        module.subscribe(&self, rhs, Box::new(Div {}))
    }
}

impl<'a> ops::Div<VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: VariableRef) -> VariableRef {
        let module = Div {};
        module.subscribe(self, &rhs, Box::new(Div {}))
    }
}

impl ops::Div<VariableRef> for VariableRef {
    type Output = VariableRef;

    fn div(self, rhs: VariableRef) -> VariableRef {
        let module = Div {};
        module.subscribe(&self, &rhs, Box::new(Div {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;

    #[test]
    fn div_check_backward() {
        let ref x = Variable::new(2.0);
        let ref y = Variable::new(3.0);

        let mut z = x / y;

        z.backward();

        assert_eq!(x.borrow().grad, 1.0 / 3.0);
        assert_eq!(y.borrow().grad, -2.0 / 9.0);
    }
}
