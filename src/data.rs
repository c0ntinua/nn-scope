use crate::vector::*;
use crate::network::*;

pub fn random_regular_datapoints( n : usize, x_range : (f64,f64) , y_range : (f64,f64)) -> Vec<(f64,f64)> {
	let mut data = vec![(0.0,0.0);n];
	let dx = (x_range.1-x_range.0)/((n+1) as f64);
	for i in 0..n {
		data[i] = ((i+1) as f64 * dx + x_range.0, random_in(y_range));
	}
	data
}

pub fn datapoints_from_function( f : fn(f64)->f64 , n : usize, x_range : (f64,f64) ) -> Vec<(f64,f64)> {
	let mut data = vec![(0.0,0.0);n];	
	for i in 0..n {
		let x = rand::random::<f64>()*(x_range.1-x_range.0) + x_range.0;
		data[i] = (x, f(x));
	}
	data
}

pub fn y_range_from_data(data : &[(f64,f64)]) -> (f64,f64) {
		let mut max_so_far = -1000000.0;
		let mut min_so_far = 1000000.0;
		for i in 0..data.len() {
			let y = data[i].1;
			if y > max_so_far { max_so_far = y;}
			if y < min_so_far { min_so_far = y;}
		}
		(min_so_far-1.0,max_so_far+1.0)
	}