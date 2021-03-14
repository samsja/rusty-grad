use std::fmt;
use std::ops;


#[derive(Debug)]
pub struct Variable {
    pub data: f32,
    grad: f32,

}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, grad : {})", self.data, self.grad)
    }
}

impl Variable {
    pub fn new(data: f32) -> Variable {
        Variable { data, grad: 0.0 }
    }
}

impl ops::Add<&Variable> for &Variable {

  type Output = Variable;

  fn add(self,_rhs: &Variable) -> Variable {
    
    Variable::new(self.data + _rhs.data)
    
  }

}
