use crate::canvas::*;
use crate::vector::*;
use crate::settings::*;
use crate::settings::Activation::*;
use crate::settings::DisplaySetting::*;
use libm::*;
use crate::num::traits::Pow;

#[derive(Debug)]
pub struct Network {
	pub rate : Vec<f64>,
	pub num_layers : usize,
	pub num_nodes : usize,
	pub value : Vec<f64>,
	pub memory : Vec<f64>,
	pub delta : Vec<f64>,
	pub bias : Vec<f64>,
	pub weight : Vec<f64>,
	pub act : Vec<Activation>,
	pub nodes_in_layer: Vec<usize>,
	pub layer_list : Vec<usize>,
	pub layer_start : Vec<usize>,
	pub layer_stop : Vec<usize>,
}


impl Network {

			
	pub fn dense(nodes_in_layer : &[usize], random_weight_span : f64) -> Network {
		let num_layers = nodes_in_layer.len(); 
		let layer_list = Network::layer_list(&nodes_in_layer);
		let layer_start = Network::layer_start(&layer_list);
		let layer_stop = Network::layer_stop(&layer_list);
		let num_nodes = layer_list.len();
		let nodes_in_layer = nodes_in_layer.to_vec();
		let mut act = vec![Relu;num_layers];
		act[0] = Identity; act[num_layers - 1] = Identity;

		let network = Network {
			rate : vec![0.00001;num_layers],
			num_layers,
			num_nodes,
			value : vec![0.0;num_nodes], 
			memory : vec![0.0;num_nodes],		
			delta : vec![0.0;num_nodes],
			bias : random_vector_in(num_nodes, (-0.00,0.00)), 
			weight : random_vector_in(num_nodes*num_nodes, (-random_weight_span,random_weight_span)), 
			act,
			nodes_in_layer,
			layer_list,
			layer_start,
			layer_stop,	
		};
		network
	}


	pub fn fwd(&mut self, x : f64) -> f64 {
		let N = self.num_nodes;
		self.value[0] = x;
		self.memory[0] = 1.0;
		for t in 1..=N-1 {
			self.value[t] = self.bias[t];
			let t_layer = self.layer_list[t];
			for s in self.layer_start[t_layer-1]..=self.layer_stop[t_layer-1] {
				self.value[t] += self.weight[t*N + s]*self.value[s];
			}
			match self.act[t_layer] {
				Identity => {self.memory[t] = 1.0;},
				Relu => {if self.value[t] > 0.0 {self.memory[t] = 1.0;} else {self.value[t] = 0.0; self.memory[t] = 0.0;}},
				Tanh => {self.value[t] = tanh(self.value[t]);self.memory[t] = 1.0 -self.value[t]*self.value[t];},
				Sin => {self.value[t] = sin(self.value[t]);self.memory[t] = cos(self.value[t]);},
				Poly => {let p = ((t - self.layer_start[t_layer])%3 + 1) as f64; 
 						self.memory[t] = p *self.value[t].powf(p-1.0);self.value[t] = self.value[t].powf(p);}, 
				_ => {print!("no act for node {}",t);},
			}
		}
		return self.value[N-1];
	}
	
	
	pub fn im_fwd(&self, x : f64) -> f64 {
		let N = self.num_nodes;
		let mut z =vec![0.0;N];
		z[0] = x;
		for t in 1..=N-1 {
			z[t] = self.bias[t];
			let t_layer = self.layer_list[t];
			for s in self.layer_start[t_layer-1]..=self.layer_stop[t_layer-1] {
				z[t] += self.weight[t*N + s]*z[s];
			}
			match self.act[t_layer] {
				Identity => (),
				Relu => {if z[t] < 0.0 {z[t] = 0.0;}}, 
				Tanh => {z[t] = tanh(z[t]);},
				Sin => {z[t] = sin(z[t]);},
 				Poly => {let p = ((t - self.layer_start[t_layer])%3 + 1) as f64;z[t] = z[t].powf(p);}
				_ => {print!("no act for node {}",t);},
			}
		}
		return z[N-1];
	}

	
	
	pub fn rtr(&mut self, x : f64) {
		let N = self.num_nodes;
		self.delta[N - 1] = x;
		for s in (0..= N - 2).rev()  {
			let s_layer = self.layer_list[s];
			self.delta[s] = 0.0;
			for t in self.layer_start[s_layer+1]..=self.layer_stop[s_layer+1] {
				self.delta[s] += self.weight[t*N + s]*self.memory[t]*self.delta[t];
			}	
		}
		for s in (1..= N - 2).rev()  {
			let s_layer = self.layer_list[s];
			self.bias[s] -= self.rate[s_layer]*self.delta[s]*self.memory[s];
			for t in self.layer_start[s_layer+1]..=self.layer_stop[s_layer+1] {
				self.weight[t*N + s] -= self.rate[s_layer]*self.memory[s]*self.delta[s]*self.value[t];
			}	
		}
	}
	
	
	pub fn clean(&mut self, weight_limit : f64) {
		for w in self.weight.iter_mut() {
			if *w > weight_limit { *w = weight_limit;}
			if *w <  -weight_limit { *w = -weight_limit;}
		}
		for b in self.bias.iter_mut() {
			if *b > weight_limit { *b = weight_limit;}
			if *b <  -weight_limit { *b = -weight_limit;}
		}
	}
	


	pub fn dense_default(layers : usize) -> Network {
		let mut v = vec![7;layers];
		v[0] = 1;
		v[layers - 1] = 1;
		Network::dense(&v, 1.0)
	}
	
		
    pub fn layer_list(nodes_in_layer : &[usize]) -> Vec<usize> {
    	let mut list = vec![];
    	for layer in 0..nodes_in_layer.len() {
    		for n in 0..nodes_in_layer[layer] { list.push(layer); }
    	}
    	list
    }
    
    pub fn positions(&self, buffer : &Canvas) -> Vec<(f32,f32)> {
    	let height = (buffer.rows-100) as f32;
    	let width  = (buffer.cols) as f32;
		let mut position =vec![ (0.0,0.0); self.layer_list.len() ];
    	let num_layers = self.nodes_in_layer.len();
    	let dx = width/((num_layers + 2) as f32);
    	let mut counter = 0;
    	for layer in 0..num_layers {
    		let dy = if self.nodes_in_layer[layer] == 1 {0.0} else { height/( self.nodes_in_layer[layer] as f32 - 1.0)};
    		let mut adjust : f32 = 0.0;
    		for n in 0..self.nodes_in_layer[layer] {
				position[counter] =((layer as f32 + 1.5) *dx, height/2.0 + adjust * dy + 50.0);
    			counter += 1;
    			adjust = if n%2 == 0 {adjust.abs() + 1.0} else {-adjust};
    		}
    	}
		position
    }
    
    pub fn layer_start(layer_list : &[usize]) -> Vec<usize> {
    	let num_nodes = layer_list.len();
    	let mut last_layer = 0;
    	let mut result = vec![];
    	for node_index in 0..num_nodes {
    		if last_layer < layer_list[node_index] { last_layer = layer_list[node_index];}
    	}
    	for layer in 0..=last_layer {
    		let mut n = 0;
    		while layer_list[n] < layer { n += 1;}
    		result.push(n);
    	}
		result
	}
	
	pub fn layer_stop(layer_list : &[usize]) -> Vec<usize> {
    	let num_nodes = layer_list.len();
    	let mut last_layer = 0;
    	let mut result = vec![];
    	for node_index in 0..num_nodes {
    		if last_layer < layer_list[node_index] { last_layer = layer_list[node_index];}
    	}
    	for layer in 0..=last_layer {
    		let mut n = 0;
    		while n < num_nodes - 1 && layer_list[n+1] <= layer { n += 1;}
    		result.push(n);
    	}
		result  
	}

	
	pub fn makeover(&mut self, nodes_in_layer : &[usize], weight_span : f64) {
		self.nodes_in_layer = nodes_in_layer.to_vec();
		self.num_layers = self.nodes_in_layer.len();
		self.layer_list = Network::layer_list(&self.nodes_in_layer);
		self.layer_start = Network::layer_start(&self.layer_list);
		self.layer_stop = Network::layer_stop(&self.layer_list);
		self.num_nodes = self.layer_list.len();
		self.act = vec![Relu;self.num_layers];
		self.act[0] = Identity; self.act[self.num_layers - 1] = Identity;
		self.value  = vec![0.0;self.num_nodes]; 
		self.memory = vec![0.0;self.num_nodes];		
		self.delta = vec![0.0;self.num_nodes];
		self.bias = random_vector_in(self.num_nodes, (-0.00,0.00)); 
		self.weight = random_vector_in(self.num_nodes*self.num_nodes, (-weight_span,weight_span)); 
		let rate = self.rate[0];
		self.rate = vec![rate;self.num_layers];
	}
	
	
		
	
						
	
}
    