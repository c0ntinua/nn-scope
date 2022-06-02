use image::{RgbImage,Pixel};
use minifb::*;
use crate::pixl::*;
use crate::buffer::*;
use crate::vector::*;
use rusttype::{Font, Scale};

pub struct Funxion <'a> {
	pub plot_buffer : Buffer,
	pub plot_rows: usize,
	pub plot_cols : usize,
	pub plot_image : RgbImage,
	pub graph_buffer : Buffer,
	pub graph_rows : usize,
	pub graph_cols : usize,
	pub graph_image : RgbImage,
	pub graph_structure : Vec<(u32,u32)>,
	pub control_buffer : Buffer,
	pub control_image : RgbImage,
	pub font : Font<'a>,
	pub machine : Vec<Machine>,
	pub delay : usize,
	pub delay_counter : usize,
	pub noise : f64,
	pub paused : bool, 
	pub help_mode : bool, 
	pub pixl_options : Vec<(usize,usize)>,
	pub current_pixl_option : usize,
	pub data: Vec<(f64,f64)>,
	pub num_data : usize,
	pub learning : bool,
}


impl Funxion<'_> {

	pub fn new(plot_rows : usize, plot_cols : usize, graph_rows : usize, graph_cols : usize) -> Funxion <'static> {
		Funxion {
			plot_buffer : Buffer::new(plot_rows, plot_cols ),
			plot_rows, 
			plot_cols, 
			plot_image : RgbImage::new(plot_cols as u32,plot_rows as u32),
			graph_buffer : Buffer::new(graph_rows, graph_cols ),
			graph_rows,
			graph_cols,
			graph_image : RgbImage::new(graph_cols as u32, graph_rows as u32),
			graph_structure : vec![],
			control_buffer : Buffer::new(graph_rows, graph_cols ),
			control_image : RgbImage::new(graph_cols as u32, graph_rows as u32),
			font : Funxion::selected_font(0),
			machine : vec![],
			delay : 0,
			delay_counter : 0,
			noise : 0.2,
			paused : false, 
			help_mode : false, 
			pixl_options : vec![(50,100)],
			current_pixl_option : 0,
			data: vec![(100.0,100.0)],
			num_data: 50,
			learning : true,
		}
	}
		
	pub fn selected_font(code : usize) -> Font <'static> {
		let font_code =  match code {
			//0 => Vec::from(include_bytes!( "cc.ttf") as &[u8]),
			0 => Vec::from(include_bytes!( "scp.ttf") as &[u8]),
			//2 => Vec::from(include_bytes!( "german.ttf") as &[u8]),
			//3 => Vec::from(include_bytes!( "sg.ttf") as &[u8]),
			//4 => Vec::from(include_bytes!( "lr.ttf") as &[u8]),
		
			_ => Vec::from(include_bytes!( "scp.ttf") as &[u8]),
		};
		Font::try_from_vec(font_code).unwrap()
	}
	pub fn plot_window(&self)  -> Window {
		Window::new(   "funxion plot", self.plot_cols, self.plot_rows, WindowOptions::default() ).expect("Unable to open Window")
}

pub fn graph_window(&self)  -> Window {
	Window::new("funxion graf", self.graph_cols, self.graph_rows, WindowOptions::default() ).expect("Unable to open Window")
}

pub fn clear_plot(&mut self) {
	self.plot_buffer.erase();
	self.plot_image = RgbImage::new(self.plot_cols as u32,self.plot_rows as u32);
}

pub fn clear_graph(&mut self) {
	self.graph_buffer.erase();
	self.graph_image = RgbImage::new(self.graph_cols as u32,self.graph_rows as u32);
}
pub fn clear_control(&mut self) {
	self.control_buffer.erase();
	self.control_image = RgbImage::new(self.graph_cols as u32,self.graph_rows as u32);
}

pub fn refresh_data(&mut self, f : fn(f64) -> f64,  x_range : (f64,f64)  ) { 
	self.data = vec![(0.0,0.0);self.num_data];
	let x  = random_vector_in(self.num_data, x_range);
	for i in 0..self.num_data  {
		let next_item = (x[i], f(x[i]) + random_in((-self.noise,self.noise)) );
		self.data[i] = next_item;
	}
}
	
		
		
	

}
	