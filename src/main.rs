
extern crate num;
extern crate libm;
extern crate ncurses;
extern crate minifb;
use minifb::*;
use libm::*;
//use image::{ImageBuffer, RgbImage, GrayImage};
//use imageproc::drawing::{draw_text_mut, text_size};
//use rusttype::{Font, Scale};
mod train;mod data;mod vector;mod escape;mod pixl;mod canvas;mod network;mod window;
use window::*;use vector::*;use train::*;use network::*;use escape::*;use canvas::*;use data::*;

fn main() {

	test();
}

fn test_learn() {

}
		
	



fn test_net() {

}




fn test() {
	
	let mut learning = true;
	let pi = std::f64::consts::PI;
	let graph_rows = 500;
	let graph_cols = 750;
	let plot_rows = 500;
	let plot_cols = 1000;
	let mut f = Network::classic(&[1,17,17,1]);
	let mut g = Network::classic(&[1,35,35,1]);
	let mut h = Network::classic(&[1,35,35,1]);
	
	let x_range = (-pi,pi);
	let mut y_range = (-20.0,20.0);
	
	let data_y_range = (-10.0,10.0);
	let batch = 1000;
	let mut data = random_regular_datapoints(10, x_range, data_y_range);
	
	let mut graph_f_canvas = Canvas::new(graph_rows,graph_cols);
	let mut graph_g_canvas = Canvas::new(graph_rows,graph_cols);
	//let mut graph_h_canvas = Canvas::new(graph_rows,graph_cols);
	let mut plot_canvas = Canvas::new(plot_rows,plot_cols);

	let mut plot_window = system_window(plot_rows, plot_cols);
	let mut f_graph_window = system_window(graph_rows, graph_cols);
	let mut g_graph_window = system_window(graph_rows, graph_cols);
	//let mut h_graph_window = system_window(graph_rows, graph_cols);
	
	plot_window.set_position(1000,0);
	f_graph_window.set_position(0,0);
	g_graph_window.set_position(0,600);
	//h_graph_window.set_position(600,600);
	plot_window.set_key_repeat_delay(0.5);
	f_graph_window.set_key_repeat_delay(0.5);
	g_graph_window.set_key_repeat_delay(0.5);
	//h_graph_window.set_key_repeat_delay(0.5);
	f.refresh_pos(&graph_f_canvas);
	g.refresh_pos(&graph_g_canvas);
	
// 	let f_closure = |x| f.im_fwd(x);
// 	let g_closure = |x| g.im_fwd(x);
// 	let h_closure = |x| h.im_fwd(x);
// 	
	while plot_window.is_open() && !plot_window.is_key_down(Key::Escape) {
		plot_canvas.clear();
		let f_closure = |x| f.im_fwd(x);
		let g_closure = |x| g.im_fwd(x);
		let h_closure = |x| h.im_fwd(x);
		plot_canvas.graphics_plot_data(&data, x_range,y_range, [255,255,255],6);
		plot_canvas.add_plot(f_closure, x_range, y_range, [255,0,0]);
		plot_canvas.add_plot(g_closure, x_range, y_range, [0,0,255]);
		//plot_canvas.add_plot(h_closure, x_range, y_range, [155,155,155]);
		plot_canvas.add_border([255,255,255]);
		
		

		
		graph_f_canvas.plot_network_weights(&f);
    	graph_f_canvas.plot_network_nodes(&f,  [255,0,0], 10);
    	graph_f_canvas.add_border([255,255,255]);
    	
    	graph_g_canvas.plot_network_weights(&g);
    	graph_g_canvas.plot_network_nodes(&g,  [0,0,255], 10);
    	graph_g_canvas.add_border([255,255,255]);
    	
    	plot_window.update_with_buffer(&plot_canvas.cells, plot_cols, plot_rows).ok();
    	f_graph_window.update_with_buffer(&graph_f_canvas.cells,graph_cols,graph_rows).ok();
		g_graph_window.update_with_buffer(&graph_g_canvas.cells,graph_cols,graph_rows).ok();
		//h_graph_window.update_with_buffer(&graph_h_canvas.cells,graph_cols,graph_rows).ok();
		
		
		plot_window.get_keys_pressed(KeyRepeat::Yes).iter().for_each(|key|
			match key {
				Key::F => {
							f = Network::random_relu_network();
							//f = Network::classic(&[1,7,7,7,7,7,1]);
							f.rate= 0.001;
						   	f.refresh_pos(&graph_f_canvas);
						   	graph_f_canvas.clear();
				},
				
				Key::G =>  {
							g = Network::random_relu_network();
							//g = Network::classic(&[1,13,13,13,1]);
							g.rate= 0.001;
						   	g.refresh_pos(&graph_g_canvas);
							graph_g_canvas.clear();
				},
				Key::H =>  {
							//h = Network::random_relu_network();
							h = Network::classic_act(&[1,27,27,27,1], &['i','r','r','r','i'] ,0.01);
							data = datapoints_from_network(&h, 20, x_range);
							y_range = y_range_from_data(&data);
						   	//h.refresh_pos(&graph_h_canvas);
							//graph_h_canvas.clear();
				},
				
				Key::L => learning = ! learning,
				
				Key::D => {
							data = datapoints_from_network(&h, 20, x_range);
							plot_canvas.clear();
							y_range = y_range_from_data(&data);
				},	
					
				_ => (),
			});

    	
		if learning {
			//train_network_with_closure(&mut f, h_closure, 100, x_range);
			//train_network_with_closure(&mut g, h_closure, 100, x_range);
			train_network_with_data(&mut f, &data, 100);
			train_network_with_data(&mut g, &data, 100);
			//h.fwd(random_in((-10.0,10.0)));
			//h.rtr(random_in((-10.0,10.0)));

		}		
	}
	}




 









