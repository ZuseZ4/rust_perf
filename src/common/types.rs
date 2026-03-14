use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct A64(pub f64);

impl Add for A64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        A64(self.0.algebraic_add(rhs.0))
    }
}
impl Sub for A64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        A64(self.0.algebraic_sub(rhs.0))
    }
}
impl Mul for A64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        A64(self.0.algebraic_mul(rhs.0))
    }
}
impl Div for A64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        A64(self.0.algebraic_div(rhs.0))
    }
}

impl AddAssign for A64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.algebraic_add(rhs.0);
    }
}

impl SubAssign for A64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.algebraic_sub(rhs.0);
    }
}

impl MulAssign for A64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = self.0.algebraic_mul(rhs.0);
    }
}

impl DivAssign for A64 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 = self.0.algebraic_div(rhs.0);
    }
}

impl A64 {
    #[inline]
    pub fn abs(self) -> A64 {
        A64(self.0.abs())
    }

    #[inline]
    pub fn sqrt(self) -> A64 {
        A64(core::f64::math::sqrt(self.0))
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0
    }
}

impl From<f64> for A64 {
    #[inline]
    fn from(val: f64) -> Self {
        A64(val)
    }
}

#[cfg(feature = "a64")]
pub type Real = A64;

#[cfg(not(feature = "a64"))]
pub type Real = f64;

#[inline]
pub const fn to_real(val: f64) -> Real {
    #[cfg(feature = "a64")]
    {
        A64(val)
    }
    #[cfg(not(feature = "a64"))]
    {
        val
    }
}

#[inline]
pub const fn from_real(val: Real) -> f64 {
    #[cfg(feature = "a64")]
    {
        val.0
    }
    #[cfg(not(feature = "a64"))]
    {
        val
    }
}

pub trait RealExt {
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
}

impl RealExt for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        core::f64::math::sqrt(self)
    }
}

impl RealExt for A64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self.0
    }
    #[inline]
    fn abs(self) -> Self {
        A64(self.0.abs())
    }
    #[inline]
    fn sqrt(self) -> Self {
        A64(core::f64::math::sqrt(self.0))
    }
}
