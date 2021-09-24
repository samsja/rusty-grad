use crate::variable::Module;
use crate::variable::VariableRef;

pub struct Relu {}

impl Module for Relu {
    fn forward(&self, x: f32, _y: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn backward<'a>(
        &self,
        grad: &'a f32,
        left_ref: &'a VariableRef,
        _right_ref: &'a VariableRef,
    ) -> [f32; 2] {
        let left_var = left_ref.borrow();

        if left_var.data > 0.0 {
            [*grad, 0.0]
        } else {
            [0.0, 0.0]
        }
    }
}

impl VariableRef {
    fn relu(&mut self) -> VariableRef {
        let module = Relu {};
        module.subscribe(self, self, Box::new(Relu {}))
    }
}

#[cfg(test)]
mod tests {

    use crate::variable::Variable;

    #[test]
    fn check_method() {
        let mut x = Variable::new(2.0);

        assert_eq!(x.relu().borrow().data, 2.0);

        let mut y = Variable::new(-10.0);

        assert_eq!(y.relu().borrow().data, 0.0);
    }

    #[test]
    fn check_backward_positive() {
        let mut x = Variable::new(2.0);
        let mut z = x.relu();
        z.backward();
        assert_eq!(x.borrow().grad, 1.0);
    }

    #[test]
    fn check_backward_negative() {
        let mut x = Variable::new(-2.0);
        let mut z = x.relu();
        z.backward();
        assert_eq!(x.borrow().grad, 0.0);
    }
}
