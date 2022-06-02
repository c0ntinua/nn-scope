
extern crate num;
extern crate libm;
extern crate ncurses;
extern crate minifb;
use minifb::*;
use libm::*;
//use image::{ImageBuffer, RgbImage, GrayImage};
//use imageproc::drawing::{draw_text_mut, text_size};
//use rusttype::{Font, Scale};
mod train;mod data;mod vector;mod escape;mod pixl;mod canvas;mod network;mod window;mod settings;
use window::*;use vector::*;use train::*;use network::*;use escape::*;use canvas::*;use data::*;
use settings::*;

fn main() {

	test();
}

fn test() {

	let font = loaded_font(0);
	let x_range = (-std::f64::consts::PI,std::f64::consts::PI);
	let y_range = (-20.0,20.0);
	let data_y_range = (-20.0,20.0);
	
	let canvas_rows = 2000;
	let canvas_cols = 2000;
	
	let graph_rows = 500;
	let graph_cols = 1000;
	
	let plot_rows = 500;
	let plot_cols = 1000;
	
	let control_rows = 1000;
	let control_cols = 500;
	
	let graph_r = 100; 
	let graph_c = 700;
	
	let plot_r = 650; 
	let plot_c = 700;
	
	let control_r = 100;
	let control_c = 100;
	

	let mut current_setting = 0;
	let mut f = Network::dense(&[1,17,17,1]);
	let mut settings = settings_from_network(&f);
	let mut learning = true;
	let mut data = random_regular_datapoints( 3, x_range, y_range);

	let mut graph_canvas = Canvas::new(graph_rows,graph_cols);
	let mut plot_canvas = Canvas::new(plot_rows,plot_cols);
	let mut control_canvas = Canvas::new(control_rows,control_cols);
	let mut canvas = Canvas::new(canvas_rows,canvas_cols);
	plot_canvas.add_plot(&f, x_range, y_range, [255,0,0]);
	//canvas.inscribe(&plot_canvas, 0,0);
	let mut canvas_window = system_window(canvas_rows, canvas_cols);
	control_canvas.add_settings(&settings, &font, current_setting);
	println!("{:?}",f.act);
	
	while canvas_window.is_open() && !canvas_window.is_key_down(Key::Escape) {
		graph_canvas.clear();
		plot_canvas.clear();
		control_canvas.clear();
		f.refresh_pos(&graph_canvas);
		
		plot_canvas.graphics_plot_data(&data, x_range,y_range, [255,255,255],6);
		graph_canvas.plot_network_weights(&f);
    	graph_canvas.plot_network_nodes(&f,  [255,0,0], 3);
    	graph_canvas.add_border([255,255,255]);
    	
		plot_canvas.add_plot(&f, x_range, y_range, [255,0,0]);
		plot_canvas.add_border([255,255,255]);
		control_canvas.add_settings(&settings, &font,current_setting);
		
		
		
		
		
		canvas.inscribe(&plot_canvas, plot_r,plot_c);
		canvas.inscribe(&graph_canvas, graph_r,graph_c);
		canvas.inscribe(&control_canvas, control_r, control_c);	
		canvas_window.update_with_buffer(&canvas.cells, canvas_rows, canvas_cols).ok();
		canvas_window.get_keys_pressed(KeyRepeat::Yes).iter().for_each(|key|
			match key {
				Key::F => 
				{
						f = Network::dense(&[1,27,31,27,1]);
						settings = settings_from_network(&f);
						f.refresh_pos(&graph_canvas);			   	
				},
				Key::X => 
				{
						f = Network::dense(&[1,27,31,27,1]);
						settings = settings_from_network(&f);
						f.refresh_pos(&graph_canvas);
						data = random_regular_datapoints( f.datapoints, x_range, data_y_range);
				},
			
				Key::L => learning = ! learning,
				
				Key::B => train_network_with_data(&mut f, &data),
				
				Key::D => 
				{
						data = random_regular_datapoints( f.datapoints, x_range, data_y_range);
				},
				Key::W => 
				{
						f.weight = random_vector_in(f.num_nodes*f.num_nodes,(-1.0,1.0));
						f.bias = random_vector_in(f.num_nodes, (-0.00,0.00) );
				}
				Key::Down => 
				{
						current_setting += 1;
						if current_setting > settings.len() - 1 {current_setting = 0;}
				},	
				Key::Up => 
				{
						if current_setting >= 1 {
							current_setting -= 1;
						} else {
							current_setting = settings.len()-1;
						}
				},
				Key::Right => 
				{
						f.respond_to_increase(&mut settings[current_setting]);
						f.refresh_pos(&graph_canvas);
						settings = settings_from_network(&f);
						control_canvas.add_settings(&settings, &font,current_setting);
				},
				Key::Left => 
				{
						f.respond_to_decrease(&mut settings[current_setting]);
						f.refresh_pos(&graph_canvas);
						settings = settings_from_network(&f);
						control_canvas.add_settings(&settings, &font,current_setting);
				},
								
				_ => (),
			});
		if learning { train_network_with_data(&mut f, &data);}
			

	}

}
		





 









