use std::ops;

use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Add {}

impl Module for Add {
    fn forward(&self, x: f32, y: f32) -> f32 {
        x + y
    }

    fn backward<'a>(
        &self,
        grad: &'a f32,
        _left_ref: &'a VariableRef,
        _right_ref: &'a VariableRef,
    ) -> [f32; 2] {
        [*grad, *grad]
    }
}

impl<'a, 'b> ops::Add<&'b VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn add(self, rhs: &'b VariableRef) -> VariableRef {
        let module = Add {};
        module.subscribe(self, rhs, Box::new(Add {}))
    }
}

impl<'a> ops::Add<&'a VariableRef> for VariableRef {
    type Output = VariableRef;

    fn add(self, rhs: &'a VariableRef) -> VariableRef {
        let module = Add {};
        module.subscribe(&self, rhs, Box::new(Add {}))
    }
}

impl<'a> ops::Add<VariableRef> for &'a VariableRef {
    type Output = VariableRef;

    fn add(self, rhs: VariableRef) -> VariableRef {
        let module = Add {};
        module.subscribe(self, &rhs, Box::new(Add {}))
    }
}

impl ops::Add<VariableRef> for VariableRef {
    type Output = VariableRef;

    fn add(self, rhs: VariableRef) -> VariableRef {
        let module = Add {};
        module.subscribe(&self, &rhs, Box::new(Add {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;

    #[test]
    fn add_check_backward() {
        let ref x = Variable::new(2.0);
        let ref y = Variable::new(3.0);

        let mut z = x + y;

        z.backward();

        assert_eq!(x.borrow().grad, 1.0);
        assert_eq!(y.borrow().grad, 1.0);
    }
}
