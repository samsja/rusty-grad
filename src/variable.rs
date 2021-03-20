use std::fmt;
use std::ops;

#[derive(Debug)]
pub struct Variable<'a> {
    pub data: f32,
    pub grad: f32,
    left_root: Option<&'a Variable<'a>>,
    right_root: Option<&'a Variable<'a>>,
}

impl<'a> fmt::Display for Variable<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Variable( {} grad : {})", self.data, self.grad)
    }
}

impl<'a> Variable<'a> {
    pub fn new_node(
        data: f32,
        left_root: Option<&'a Variable<'a>>,
        right_root: Option<&'a Variable<'a>>,
    ) -> Variable<'a> {
        Variable {
            data,
            grad: 0.0,
            left_root,
            right_root,
        }
    }
}

impl<'a> Variable<'a> {
    // TODO new should be private
    pub fn new(data: f32) -> Variable<'a> {
        Variable::new_node(data, None, None)
    }
}

impl<'a> ops::Add<&'a Variable<'a>> for &'a Variable<'a> {
    type Output = Variable<'a>;

    fn add(self, rhs: &'a Variable<'a>) -> Variable<'a> {
        Variable::new_node(self.data + rhs.data, Some(&self), Some(&rhs))
    }
}
