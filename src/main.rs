
extern crate num;
extern crate libm;
extern crate ncurses;
extern crate minifb;
use minifb::*;
use libm::*;
//use image::{ImageBuffer, RgbImage, GrayImage};
//use imageproc::drawing::{draw_text_mut, text_size};
//use rusttype::{Font, Scale};
mod train;mod data;mod vector;mod escape;mod canvas;mod network;mod window;mod settings;
use window::*;use vector::*;use train::*;use network::*;use escape::*;use canvas::*;use data::*;
use settings::*;

fn main() {
	let mut display_error = 0.0;
	let font = loaded_font(1);
	let function_plot_thickness = 2;
	let mut data_mode = false;
	let mut f = Network::dense(&[1,17,17,1]);
	let x_range = (f.x_min,f.x_max);
	let y_range = (f.y_min,f.y_max);
	let data_y_range = (-10.0,10.0);
	
	let canvas_rows = 2000;
	let canvas_cols = 2000;
	
	let graph_rows = 500;
	let graph_cols = 1300;
	
	let plot_rows = 500;
	let plot_cols = 1000;
	
	let control_rows = 1100;
	let control_cols = 300;
	
	let graph_r = 10; 
	let graph_c = 300;
	
	let plot_r = 510; 
	let plot_c = 400;
	
	let control_r = 10;
	let control_c = 10;
	

	let mut current_setting = 0;
	let mut f = Network::dense(&[1,17,17,17,17,17,1]);
	let g = libm::sin;
	let g = |x| 8.0*sin(2.0*x);
	let mut settings = settings_from_network(&f);
	let mut learning = true;
	let mut data = random_regular_datapoints( 3, x_range, y_range);

	let mut graph_canvas = Canvas::new(graph_rows,graph_cols);
	let mut plot_canvas = Canvas::new(plot_rows,plot_cols);
	let mut control_canvas = Canvas::new(control_rows,control_cols);
	let mut canvas = Canvas::new(canvas_rows,canvas_cols);
	//canvas.inscribe(&plot_canvas, 0,0);
	let mut canvas_window = system_window(canvas_rows, canvas_cols);
	control_canvas.add_settings(&settings, &font, current_setting);
	println!("{:?}",f.act);
	
	while canvas_window.is_open() && !canvas_window.is_key_down(Key::Escape) {
		let x_range = (f.x_min,f.x_max);let y_range = (f.y_min,f.y_max);
		f.refresh_pos(&graph_canvas);
		graph_canvas.clear();plot_canvas.clear();control_canvas.clear();
		
		let f_clo = |x| f.im_fwd(x);
		
		if data_mode {
			plot_canvas.add_data(&data, x_range,y_range, [255,255,0],8);
		} else {
			plot_canvas.add_closure_t(g, x_range, y_range, [0,0,255],function_plot_thickness);
		}
			
		graph_canvas.add_network_weights(&f);
    	graph_canvas.add_network_nodes(&f,  [255,0,0], 6);
    	graph_canvas.load_cells_from_image();
    	
    	plot_canvas.add_grid(x_range, y_range, [255,255,255]);
		plot_canvas.add_error(display_error, &font, 0,0, [155,155,155]);
		plot_canvas.add_closure_t(f_clo, x_range, y_range, [255,0,0],function_plot_thickness);
		
		plot_canvas.load_cells_from_image();
		
		control_canvas.add_settings(&settings, &font,current_setting);
		control_canvas.load_cells_from_image();
		
		canvas.inscribe(&plot_canvas, plot_r,plot_c);
		canvas.inscribe(&graph_canvas, graph_r,graph_c);
		canvas.inscribe(&control_canvas, control_r, control_c);	
		
		canvas_window.update_with_buffer(&canvas.cells, canvas_rows, canvas_cols).ok();
		canvas_window.get_keys_pressed(KeyRepeat::Yes).iter().for_each(|key|
			match key {
				Key::F => 
				{
						f.makeover(&[1,27,31,27,1]);
						settings = settings_from_network(&f);
						f.refresh_pos(&graph_canvas);			   	
				},
				Key::X => 
				{
						f.makeover(&[1,11,9,13,9,11,1]);
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
				Key::M => data_mode = !data_mode,
								
				_ => (),
			});
		match learning { 
			true => for _ in 0..f.batch_size {
						display_error = match data_mode {
							true => train_network_with_data1(&mut f, &data),
							false => train_network_with_function1(&mut f, g, x_range),
						}
					},
			false => (),
		}
	}
}







 









