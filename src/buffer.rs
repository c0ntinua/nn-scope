use crate::pixl::*;
use crate::network::*;
use image::{RgbImage,ImageBuffer, Pixel};
use imageproc::drawing::*;
use rand::random;


pub struct Buffer {
	pub rows : usize,
	pub cols : usize,
	pub pixl : Pixl,
	pub cells : Vec<u32>,
	pub radius : i32,
}

impl Buffer {
	pub fn new(rows : usize, cols : usize) -> Buffer {
		Buffer {
			rows,
			cols,
			pixl : Pixl::default_pixl(1,1,rows,cols),
			cells : vec![0u32;rows*cols],
			radius: 8,
		}
	}
	pub fn add_border(&mut self, rgb : [u32;3]) {
		let rows = self.pixl.rows;
		let cols = self.pixl.cols;
		for col in 0..cols {
			self.draw_pixl(0, col, rgb);
			self.draw_pixl(rows -1, col, rgb);
		}
		for row in 0..rows {
			self.draw_pixl(row, 0, rgb);
			self.draw_pixl(row, cols-1, rgb);
		}
	}
	pub fn add_guide(&mut self,x_range : (f64,f64) , y_range : (f64,f64) , rgb : [u32;3]) {
		let rows = self.pixl.rows;
		let cols = self.pixl.cols;
		let unit_mark = 0;
		let row = rows/2;
		for col in 0..cols {
			//self.cells[row*cols+col] = rgb[0] << 16 | rgb[1] << 8 | rgb[2];
			self.draw_pixl(row, col, rgb);
		}
		let col = cols/2;
		for row in 0..rows {
			//self.cells[row*cols+col] = rgb[0] << 16 | rgb[1] << 8 | rgb[2];
			self.draw_pixl(row, col, rgb);
		}
	}
			
		
		
			
		

	pub fn add_plot<F>(&mut self, mut f : F, x_range : (f64,f64), y_range : (f64,f64), rgb : [u32;3] )
		where F: Fn(f64) -> f64 {
		let rows = self.pixl.rows;
		let cols = self.pixl.cols;
		let pixels_per_unit_x = cols as f64 / (x_range.1 - x_range.0);
		let x_unit_per_col = (x_range.1 - x_range.0)/(cols as f64);
		let pixels_per_unit_y = rows as f64 / (y_range.1 - y_range.0);
		let mut current_x = x_range.0;
		for col in 0..cols {
			current_x = col as f64 * x_unit_per_col + x_range.0;
			let f_of_x = (f)(current_x);
			if f_of_x > y_range.0 && f_of_x < y_range.1 {
				let row = (( (f_of_x - y_range.0)/(y_range.1 - y_range.0) )*(rows as f64)).floor() as usize;
				self.draw_pixl(row, col, rgb);
				self.draw_pixl(row + 1, col, rgb);
				self.draw_pixl(row , col + 1, rgb);
				self.draw_pixl(row + 1, col + 1, rgb);
			} 
		}		
	}
	
	pub fn rgb_plot<F>( &mut self, mut f : F, im : &mut RgbImage, x_range : (f64,f64), y_range : (f64,f64), rgb : [u8;3])
		where F: Fn(f64) -> f64 {
		//let mut im = RgbImage::new(self.cols as u32,self.rows as u32);
		let rows = self.pixl.rows;
		let cols = self.pixl.cols;
		let pixels_per_unit_x = cols as f64 / (x_range.1 - x_range.0);
		let x_unit_per_col = (x_range.1 - x_range.0)/(cols as f64);
		let pixels_per_unit_y = rows as f64 / (y_range.1 - y_range.0);
		let mut current_x = x_range.0;
		for col in 0..cols {
			current_x = col as f64 * x_unit_per_col + x_range.0;
			let f_of_x = (f)(current_x);
			if f_of_x > y_range.0 && f_of_x < y_range.1 {
				let row = (( (f_of_x - y_range.0)/(y_range.1 - y_range.0) )*(rows as f64)).floor() as usize;
				self.draw_rgb_pixl(im, row, col, rgb);
			} 
		}
		self.cells = Buffer::as_u32_buffer(&im);
	}
	
	pub fn graphics_plot<F>( &mut self, mut f : F, im : &mut RgbImage, x_range : (f64,f64), y_range : (f64,f64), rgb : [u8;3])
		where F: Fn(f64) -> f64 {
		//let mut im = RgbImage::new(self.cols as u32,self.rows as u32);
		let rows = self.pixl.rows;
		let cols = self.pixl.cols;
		let pixels_per_unit_x = cols as f64 / (x_range.1 - x_range.0);
		let x_unit_per_col = (x_range.1 - x_range.0)/(cols as f64);
		let pixels_per_unit_y = rows as f64 / (y_range.1 - y_range.0);
		let mut current_x = x_range.0;
		for col in 0..cols {
			current_x = col as f64 * x_unit_per_col + x_range.0;
			let f_of_x = (f)(current_x);
			if f_of_x > y_range.0 && f_of_x < y_range.1 {
				let row = (( (f_of_x - y_range.0)/(y_range.1 - y_range.0) )*(rows as f64)).floor() as usize;
				self.draw_rgb_circle(im, row, col, rgb);
				
			} 
		}
		self.cells = Buffer::as_u32_buffer(&im);
	}
	
	pub fn graphics_plot_data( &mut self, data : &[(f64,f64)], im : &mut RgbImage, x_range : (f64,f64), y_range : (f64,f64), rgb : [u8;3]) {
		let rows = self.pixl.rows;
		let cols = self.pixl.cols;
		let pixels_per_unit_x = cols as f64 / (x_range.1 - x_range.0);
		let x_unit_per_col = (x_range.1 - x_range.0)/(cols as f64);
		let pixels_per_unit_y = rows as f64 / (y_range.1 - y_range.0);
		let mut current_x = x_range.0;
		for i in 0..data.len() {
			if data[i].0 > x_range.0 && data[i].0  < x_range.1 {
				if data[i].1 > y_range.0 && data[i].1  < y_range.1 {
					let col = (pixels_per_unit_x *(data[i].0 - x_range.0)) .floor() as usize;
					let row = (pixels_per_unit_y *(data[i].1 - y_range.0)) .floor() as usize;
					self.draw_rgb_circle(im, row , col , rgb); 
				}
			} 
		}
		self.cells = Buffer::as_u32_buffer(&im);
	}
	
	pub fn add_data(&mut self, data : &[(f64,f64)], x_range : (f64,f64), y_range : (f64,f64), rgb : [u32;3] ) {
		let cols = self.pixl.cols;
		let rows = self.pixl.rows;
		let pixels_per_unit_x = cols as f64 / (x_range.1 - x_range.0);
		let x_unit_per_col = (x_range.1 - x_range.0)/(cols as f64);
		let pixels_per_unit_y = rows as f64 / (y_range.1 - y_range.0);
		let mut current_x = x_range.0;
		for i in 0..data.len() {
			if data[i].0 > x_range.0 && data[i].0  < x_range.1 {
				if data[i].1 > y_range.0 && data[i].1  < y_range.1 {
					let col = (pixels_per_unit_x *(data[i].0 - x_range.0)) .floor() as usize;
					let row = (pixels_per_unit_y *(data[i].1 - y_range.0)) .floor() as usize;
					self.draw_pixl(row, col, rgb);
					//self.draw_rgb_circle(im, row , col , rgb); 
				}
			} 
		}
	}
			
	pub fn draw_rgb_pixl(&mut self, im : &mut RgbImage, r : usize, c : usize,  rgb : [u8;3]) {
		let buffer_cols = self.cols;
		let pixl_height = self.pixl.height;
		let pixl_width = self.pixl.width;		
		let start_col = c*pixl_width;
		let start_row = r*pixl_height;
		for sub_row in 0..pixl_height {
			for sub_col in 0..pixl_width {
				let im_col = (start_col + sub_col) as u32;
				let im_row = (start_row + sub_row) as u32;
				*im.get_pixel_mut(im_col, im_row) = image::Rgb(rgb);
			}
		}			
	}
	
	pub fn draw_rgb_circle(&mut self, im : &mut RgbImage, r : usize, c : usize,  rgb : [u8;3]) {
		let buffer_cols = self.cols;
		let pixl_height = self.pixl.height;
		let pixl_width = self.pixl.width;		
		let col = c*pixl_width + pixl_width/2;
		let row = r*pixl_height + pixl_height/2;
		draw_filled_circle_mut(im, (col as i32, row as i32),self.radius, image::Rgb(rgb)); 			
	}
	
	
	pub fn draw_pixl(&mut self, r : usize, c : usize,  rgb : [u32;3]) {
		if r > self.pixl.rows - 1  {return;}
		if c > self.pixl.cols - 1  {return;}
		let buffer_cols = self.cols;
		let pixl_height = self.pixl.height;
		let pixl_width = self.pixl.width;		
		let start_col = c*pixl_width;
		let start_row = r*pixl_height;
		for sub_row in 0..pixl_height {
			for sub_col in 0..pixl_width {
				let index = (start_row + sub_row)*buffer_cols + start_col + sub_col;
				self.cells[index] = rgb[0] << 16 | rgb[1] << 8 | rgb[2];
			}
		}
					
	}
	
	pub fn as_u32_buffer(im : &RgbImage) -> Vec<u32> {
		let rows = im.height() as usize; 
		let cols = im.width() as usize; 
		let mut buffer = vec![0u32;rows*cols];
		for row in 0..rows {
			for col in 0..cols{
				let rgb = im.get_pixel(col as u32,row as u32).channels();
				let r = rgb[0] as u32;
				let g = rgb[1] as u32;
				let b = rgb[2] as u32;
				let code = r << 16 | g << 8 | b;
				buffer[row*cols+col] = code;
			}
		}	
		buffer
	}
	
	pub fn erase(&mut self) {
		self.cells = vec![0u32;self.rows*self.cols];
	}
	

    pub fn install_structure(&mut self, node : &[usize], height : usize, width : usize) -> Vec<(usize,usize)> {
    	let mut node_position : Vec<(usize,usize)> = vec![(0,0);node.len()];
    	let num_layers = node.len();
    	let dx = width/(num_layers + 1);
    	let mut counter = 0;
    	for layer in 0..num_layers {
    		let dy = height/node[layer] - 10;
    		let mut adjust : i64 = 0;
    		for n in 0..node[layer] {
    			node_position[counter] = (layer*dx, height/2 + (adjust * dy as i64) as usize);
    			counter += 1;
    			adjust = if n%2 == 0 {adjust.abs() + 1} else {-adjust};
    		}
    	}
    	node_position
    }
    

    pub fn plot_network_weights(&mut self, network : &Network, im : &mut RgbImage) {
    	let N = network.layer_list.len();
    	for s in 0..=N-2 {
			let s_layer = network.layer_list[s];
			for t in network.layer_start[s_layer+1]..=network.layer_stop[s_layer+1] {
				let hue = ((network.weight[t*N+s].abs()/network.weight_limit)*255.0).trunc() as u8;
				let rgb = [hue,hue,hue];
				draw_line_segment_mut( im, (network.pos_x[s], network.pos_y[s]), (network.pos_x[t], network.pos_y[t]), image::Rgb(rgb));
			}
		}
		self.cells = Buffer::as_u32_buffer(&im);	
	}
	
	pub fn plot_network_nodes(&mut self, network : &Network, im : &mut RgbImage, rgb : [u8;3]) {
    	let num_nodes = network.layer_list.len();
		for n in 0..num_nodes {
			draw_filled_circle_mut(im, (network.pos_x[n].trunc() as i32, network.pos_y[n].trunc() as i32) , self.radius, image::Rgb(rgb)); 	
		}
		self.cells = Buffer::as_u32_buffer(&im);	
	}
				
    			
        
}
		
	

