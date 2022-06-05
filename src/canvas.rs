
use crate::network::*;
use image::{RgbImage, Pixel};
use rusttype::{Font, Scale};
use imageproc::drawing::*;
use rand::random;
use crate::settings::*;
use crate::settings::DisplaySetting::*;

pub struct Canvas {
	pub rows : usize,
	pub cols : usize,
	pub cells : Vec<u32>,
	pub image : RgbImage,
}

impl Canvas {

	pub fn new(rows : usize, cols : usize) -> Canvas {
			Canvas {
				rows,
				cols,
				cells : vec![0u32;rows*cols],
				image: RgbImage::new(cols as u32,rows as u32),
			}
	}
	
	pub fn clear(&mut self) {
		self.cells = vec![0u32;self.rows*self.cols];
		self.image = RgbImage::new(self.cols as u32,self.rows as u32);
	}
	
	pub fn load_cells_from_image(&mut self) {
		self.cells = Canvas::as_u32_buffer(&self.image);
	}		
	
	
	pub fn inscribe(&mut self, source : &Canvas, r : usize, c : usize) {
		let source_cols = source.cols;
		let self_cols = self.cols;
		for row in 0..source.rows {
			for col in 0..source_cols {
				self.cells[(row + r)*self_cols + col + c] = source.cells[row*source_cols+col];
			}
		}
	}
	

	pub fn add_grid(&mut self, x_range : (f64,f64), y_range : (f64,f64),rgb : [u32;3]) {
		let dim_value = [20,20,20];
		let bright_value = [50,50,50];
		let height = self.image.height();
		let width = self.image.width();
		let mut current_val = y_range.0.trunc();
		while  current_val <= y_range.1 {
			let row = ((current_val -  y_range.0)/(y_range.1 - y_range.0)) as f32;
			let row = row * height as f32;
			let this_rgb = if current_val == 0.0 {bright_value} else {dim_value};
			draw_line_segment_mut(&mut self.image, 
				(0.0, row ), ((width - 1) as f32, row), image::Rgb(this_rgb));
			current_val += 1.0;
		}
		let mut current_val = x_range.0.trunc();
		while  current_val <= x_range.1 {
			let col = ((current_val -  x_range.0)/(x_range.1 - x_range.0)) as f32;
			let col = col * width as f32;
			let this_rgb = if current_val == 0.0 {bright_value} else {dim_value};
			draw_line_segment_mut(&mut self.image, 
				(col, 0.0), (col , (height - 1) as f32), image::Rgb(this_rgb));
			current_val += 1.0;
		}
	}
		
	pub fn add_closure<F>(&mut self, g : F, x_range : (f64,f64), y_range : (f64,f64), rgb : [u8;3], thickness : usize ) 
		where F : Fn(f64) -> f64 {
		let rows = self.rows;
		let cols = self.cols;
		let mut last_col = 0f32;
		let mut last_row = 0f32;
		
		let x_unit_per_col = (x_range.1 - x_range.0)/(cols as f64);
		let mut current_x = x_range.0;
		for col in 0..cols {
			current_x = col as f64 * x_unit_per_col + x_range.0;
			let f_of_x = g(current_x);
			if f_of_x > y_range.0 && f_of_x < y_range.1 {
				let this_row = (( (f_of_x - y_range.0)/(y_range.1 - y_range.0) )*(rows as f64)).floor() as f32;
				let this_col = col as f32;
				if col != 0 {
						Canvas::draw_thick_line(&mut self.image, (last_col, last_row), (this_col, this_row), thickness, rgb);
					}
				last_col = this_col; last_row= this_row;
				//draw_filled_circle_mut(&mut self.image, (this_col, this_row ) , radius, image::Rgb(rgb));

			} 
		}

	}
	pub fn draw_thick_line(image : &mut RgbImage, start : (f32,f32), stop : (f32,f32), thickness : usize, rgb : [u8;3]) {
		let piece = 1.0;
		let thickness =  thickness as i64;
		for i in (-thickness)..thickness {
			let mod_start = (start.0 + (i as f32)*piece, start.1 + (i as f32)*piece);
			let mod_stop = (stop.0 + (i as f32)*piece, stop.1 + (i as f32)*piece);
			draw_line_segment_mut(image, mod_start, mod_stop, image::Rgb(rgb));
		}
	}
			
		
	
	pub fn add_data( &mut self, data : &[(f64,f64)], x_range : (f64,f64), y_range : (f64,f64), 
		rgb : [u8;3], radius : i32) {
		let rows = self.rows;
		let cols = self.cols;
		let pixels_per_unit_x = cols as f64 / (x_range.1 - x_range.0);
		let x_unit_per_col = (x_range.1 - x_range.0)/(cols as f64);
		let pixels_per_unit_y = rows as f64 / (y_range.1 - y_range.0);
		let mut current_x = x_range.0;
		for i in 0..data.len() {
			if data[i].0 > x_range.0 && data[i].0  < x_range.1 {
				if data[i].1 > y_range.0 && data[i].1  < y_range.1 {
					let col = (pixels_per_unit_x *(data[i].0 - x_range.0)) .floor() as i32;
					let row = (pixels_per_unit_y *(data[i].1 - y_range.0)) .floor() as i32;
					draw_filled_circle_mut(&mut self.image, (col, row ) , radius, image::Rgb(rgb)); 
				}
			} 
		}
	}
	pub fn add_network_weights(&mut self, network : &Network, pos : &[(f32,f32)], weight_limit : f64) {
    	let N = network.layer_list.len();
    	for s in 0..=N-2 {
			let s_layer = network.layer_list[s];
			for t in network.layer_start[s_layer+1]..=network.layer_stop[s_layer+1] {
				let hue = ((network.weight[t*N+s].abs()/weight_limit)*255.0).trunc() as u8;
				//let hue = if hue > 128 { hue} else {0};
				let hue = if hue < 128 {0} else {2*(hue-128) };
				let rgb = [hue,hue,hue];
				//let thickness = (hue / 64) as usize;
				let thickness = 1;
				Canvas::draw_thick_line(&mut self.image, pos[s] , 
					pos[t], thickness, rgb); 
			}
		}
	}
	
	pub fn add_network_nodes(&mut self, pos : &[(f32,f32)], rgb : [u8;3], radius : i32) {
    	let num_nodes = pos.len();
		for n in 0..num_nodes {
			draw_filled_circle_mut(&mut self.image, (pos[n].0.trunc() as i32, pos[n].1.trunc()as i32) , radius, image::Rgb(rgb)); 	
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
	

	pub fn add_settings(&mut self, settings : &[DisplaySetting], font : &Font, current_setting :  usize) {
		let size_float = 20.0; 
		let size_int = 20;
		let scale = Scale { x: size_float, y: size_float};
		let mut rgb = [255u8;3];
		let mut display_string = vec!["not_implemented_yet".to_string();settings.len()];
		for i in 0..settings.len() {
			display_string[i] = match settings[i] {
				Rate(r) =>                                   format!("rate   =  {:.6}",r),
				NumLayers(n) =>                              format!("layers =  {:03}",n),
				NodesInLayer{num_nodes : n, layer : l} =>    format!("lay{:2}  =  {:03}",l,n),
				ActivationOfLayer{ act : f ,layer : l }=>    format!("fun{:2}  =  {}",l,f.abbr()),
				RateOfLayer{ rate: r, layer : l} =>          format!("rate{:1}    =  {:.5}",l,r),
				WeightLimit(w) => 						     format!("wgtlmt =  {:.1}",w), 
				BatchSize(n) =>                              format!("btch   =  {:03}",n),
				Datapoints(n) =>                             format!("data   =  {:03}",n),
				XMin(m) => 									 format!("fxmin  =  {:+03}",m.trunc() as i32),
				XMax(m) => 									 format!("fxmax  =  {:+03}",m.trunc() as i32),
				YMin(m) => 									 format!("fymin  =  {:+03}",m.trunc() as i32),
				YMax(m) => 									 format!("fymax  =  {:+03}",m.trunc() as i32),
				DataXMin(m) => 							     format!("dxmin  =  {:+03}",m.trunc() as i32),
				DataXMax(m) => 							     format!("dxmax  =  {:+03}",m.trunc() as i32),
				DataYMin(m) => 							     format!("dymin  =  {:+03}",m.trunc() as i32),
				DataYMax(m) => 							     format!("dymax  =  {:+03}",m.trunc() as i32),
				_ => format!("not implemented yet"),
			}
		}
		for i in 0..settings.len() {
			let j = (i%2) as i32;
			let x = 10;
			let y = i as i32 * size_int;
			if i == current_setting { rgb = [255u8,255u8,0u8];} else { rgb =[255u8,255u8,255u8];}
			draw_text_mut(& mut self.image, image::Rgb(rgb), x, y, scale, font,  &display_string[i]);
		}
	}
	pub fn add_error(&mut self, error : f64, font : &Font, row : i32, col : i32, rgb : [u8;3]) {
		let size_float = 40.0; 
		let scale = Scale { x: size_float, y: size_float};
		let display_string = format!("{:.10}",error);
		draw_text_mut(& mut self.image, image::Rgb(rgb), col, row, scale, font,  &display_string);
	}
		
	
}