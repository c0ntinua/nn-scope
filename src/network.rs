use crate::canvas::*;
use crate::vector::*;
use crate::settings::*;
use crate::settings::Activation::*;
use crate::settings::Setting::*;
use libm::*;
use crate::num::traits::Pow;

#[derive(Debug)]
pub struct Network {
	pub rate : Vec<f64>,
	pub num_nodes : usize,
	pub num_layers : usize,
	pub value : Vec<f64>,
	pub memory : Vec<f64>,
	pub last_out : Vec<f64>,
	pub delta : Vec<f64>,
	pub bias : Vec<f64>,
	pub weight : Vec<f64>,
	pub act : Vec<Activation>,
	pub nodes_in_layer: Vec<usize>,
	pub layer_list : Vec<usize>,
	pub layer_start : Vec<usize>,
	pub layer_stop : Vec<usize>,
	pub weight_limit : f64,
	pub pos_x: Vec<f32>,
	pub pos_y: Vec<f32>,
	pub current_setting : usize,
	pub batch_size : usize,
	pub datapoints : usize,
}

impl Network {
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
				// 'p' => {let p = ((t - self.layer_start[t_layer])%3 + 1) as f64; 
// 						self.memory[t] = p *self.value[t].powf(p-1.0);self.value[t] = self.value[t].powf(p);}, 
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
// 				'p' => {let p = ((t - self.layer_start[t_layer])%3 + 1) as f64;z[t] = z[t].powf(p);}
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
				//println!("add {:.20} to weight to {:2} from {:2} ", self.rate*self.memory[s]*self.delta[s]*self.value[t], t,s);
				self.weight[t*N + s] -= self.rate[s_layer]*self.memory[s]*self.delta[s]*self.value[t];
				//println!("new weight = {:.6} ",self.weight[t*N + s] );
			}	
		}
	}
	
	
	pub fn clean(&mut self) {
		for w in self.weight.iter_mut() {
			if *w > self.weight_limit { *w = self.weight_limit;}
			if *w <  -self.weight_limit { *w = -self.weight_limit;}
		}
		for b in self.bias.iter_mut() {
			if *b > self.weight_limit { *b = self.weight_limit;}
			if *b <  -self.weight_limit { *b = -self.weight_limit;}
		}
	}
	

			
	pub fn dense(nodes_in_layer : &[usize]) -> Network {
		let num_layers = nodes_in_layer.len(); 
		let layer_list = Network::layer_list(&nodes_in_layer);
		let layer_start = Network::layer_start(&layer_list);
		let layer_stop = Network::layer_stop(&layer_list);
		let num_nodes = layer_list.len();
		let nodes_in_layer = nodes_in_layer.to_vec();
		let mut act = vec![Relu;num_layers];
		act[0] = Identity; act[num_layers - 1] = Identity;
		let network = Network {
			rate : vec![0.0001;num_layers],
			num_nodes,
			num_layers,
			value : vec![0.0;num_nodes], 
			memory : vec![0.0;num_nodes],
			last_out : vec![0.0;num_nodes],
			delta : vec![0.0;num_nodes],
			bias : random_vector_in(num_nodes, (-0.00,0.00)), 
			weight : random_vector_in(num_nodes*num_nodes, (-1.0,1.0)), 
			act,
			nodes_in_layer,
			layer_list,
			layer_start,
			layer_stop,
			weight_limit : 5.0,
			pos_x : vec![0.0;num_nodes],
			pos_y : vec![0.0;num_nodes],
			current_setting : 0,
			batch_size : 100,
			datapoints : 5,
		};
		network
	}
	pub fn dense_default(layers : usize) -> Network {
		let mut v = vec![7;layers];
		v[0] = 1;
		v[layers - 1] = 1;
		Network::dense(&v)
	}
	
		
    pub fn layer_list(nodes_in_layer : &[usize]) -> Vec<usize> {
    	let mut list = vec![];
    	for layer in 0..nodes_in_layer.len() {
    		for n in 0..nodes_in_layer[layer] { list.push(layer); }
    	}
    	list
    }
    
    pub fn refresh_pos(&mut self, buffer : &Canvas) {
    	let height = (buffer.rows-100) as f32;
    	let width  = (buffer.cols) as f32;
    	self.pos_x = vec![0.0;self.layer_list.len()];
    	self.pos_y = vec![0.0;self.layer_list.len()];
    	let num_layers = self.nodes_in_layer.len();
    	let dx = width/((num_layers + 2 ) as f32);
    	let mut counter = 0;
    	for layer in 0..num_layers {
    		let dy = if self.nodes_in_layer[layer] == 1 {0.0} else { height/( self.nodes_in_layer[layer] as f32 - 1.0)};
    		let mut adjust : f32 = 0.0;
    		for n in 0..self.nodes_in_layer[layer] {
    			self.pos_x[counter] = ((layer as f32 + 1.5) *dx);
    			self.pos_y[counter] = height/2.0 + adjust * dy + 50.0;
    			counter += 1;
    			adjust = if n%2 == 0 {adjust.abs() + 1.0} else {-adjust};
    		}
    	}
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
	
	pub fn estimated_y_range(&self, x_range : (f64,f64)) -> (f64,f64) {
		let mut max_so_far = -1000000.0;
		let mut min_so_far = 1000000.0;
		let dx = (x_range.1-x_range.0)/100.0;
		for i in 0..100 {
			let sample_x = x_range.0 + (i as f64)*dx;
			let y = self.im_fwd(sample_x);
			if y > max_so_far { max_so_far = y;}
			if y < min_so_far { min_so_far = y;}
		}
		(min_so_far-1.0,max_so_far+1.0)
	}
	pub fn makeover(&mut self, nodes_in_layer : &[usize]) {
		self.nodes_in_layer = nodes_in_layer.to_vec();
		self.num_layers = self.nodes_in_layer.len(); 
		self.layer_list = Network::layer_list(&self.nodes_in_layer);
		self.layer_start = Network::layer_start(&self.layer_list);
		self.layer_stop = Network::layer_stop(&self.layer_list);
		self.num_nodes = self.layer_list.len();
		self.act = vec![Relu;self.num_layers];
		self.act[0] = Identity; self.act[self.num_layers - 1] = Identity;
		self.weight = 	random_vector_in(self.num_nodes*self.num_nodes, (-1.0,1.0)  );
		self.bias = random_vector_in(self.num_nodes, (-0.00,0.00) ); 
		self.pos_x = vec![0.0;self.num_nodes];
		self.pos_y = vec![0.0;self.num_nodes];
		self.value = vec![0.0;self.num_nodes];
		self.memory = vec![0.0;self.num_nodes];
		self.delta = vec![0.0;self.num_nodes];
		self.rate = vec![0.0001;self.num_layers];
		self.weight_limit = 5.0;
	}
	
	
	pub fn respond_to_increase(&mut self, setting : &mut Setting) {
		match setting {
				&mut Setting::NumLayers(n) => {
					if n < 9 {
						*setting = NumLayers(n+1);
						let mut next = vec![13;n+1];
						next[0] = 1;
						next[n] = 1;
						self.makeover(&next);
						}
				},
				&mut Setting::Rate(r) => {  
					*setting = Rate(r+0.0001);
					self.rate = vec![r+0.001;self.num_layers];
				},
				&mut Setting::WeightLimit(l) => {  
					*setting = WeightLimit(l+1.0);
					self.weight_limit = l + 1.0;
				},
				&mut Setting::BatchSize(s) => {  
					*setting = BatchSize(s+1);
					self.batch_size= s + 1;
				},
				&mut Setting::Datapoints(s) => {
					if self.datapoints < 100 {
						*setting = Datapoints(s+1);
						self.datapoints= s +1;
					}
				},
				&mut Setting::NodesInLayer{num_nodes :n , layer : l} => {
					if self.nodes_in_layer[l] <= 57 && l != 0 && l != self.num_layers - 1 {
						*setting = NodesInLayer { num_nodes : n+2 , layer :l };
						self.nodes_in_layer[l] += 2;
						let next = self.nodes_in_layer.clone();
						self.makeover(&next);
					}
				},
				&mut Setting::ActivationOfLayer{act : f , layer : l} => {
					let next = match f {
						Identity => Relu,
						Relu => Tanh,
						Tanh => Sin,
						Sin => Identity,
					};
					*setting = ActivationOfLayer{act : next , layer : l};
					self.act[l] = next;
				},
				_ => (),
				}
												
// 												
// 				NodesInLayer{num_nodes : n, layer : l} =>    format!("nodes{:1}    =  {:02}",l,n),
// 				ActivationOfLayer{ act : f ,layer : l }=>    format!("funct{:1}    =  {}",l,f.abbr()),
// 				RateOfLayer{ rate: r, layer : l} =>          format!("rate{:1}     =  {:.4}",l,r),
// 				WeightLimit(w) => 						     format!("wghtlmt   =  {:.1}",w), 
// 				BatchSize(n) =>                              format!("batch     =  {:03}",n),
// 				_ => format!("not implemented yet"),
	}
	pub fn respond_to_decrease(&mut self, setting : &mut Setting) {
		match setting {
				&mut Setting::NumLayers(n) => {
					if n > 3 {
						*setting = NumLayers(n-1);
						self.nodes_in_layer = vec![7;n-1];
						self.nodes_in_layer[0] = 1;
						self.nodes_in_layer[n-2] = 1;
						self.num_layers = self.nodes_in_layer.len(); 
						self.layer_list = Network::layer_list(&self.nodes_in_layer);
						self.layer_start = Network::layer_start(&self.layer_list);
						self.layer_stop = Network::layer_stop(&self.layer_list);
						self.num_nodes = self.layer_list.len();
						self.act = vec![Relu;self.num_layers];
						self.act[0] = Identity; self.act[self.num_layers - 1] = Identity;
						self.weight = 	random_vector_in(self.num_nodes*self.num_nodes, (-1.0,1.0)  );
						self.bias = random_vector_in(self.num_nodes, (-0.00,0.00) ); 
						self.pos_x = vec![0.0;self.num_nodes];
						self.pos_y = vec![0.0;self.num_nodes];
						self.value = vec![0.0;self.num_nodes];
						self.memory = vec![0.0;self.num_nodes];
						self.delta = vec![0.0;self.num_nodes];
						self.rate = vec![0.001;self.num_layers];
					}
				},
				&mut Setting::Rate(r) => {  
					*setting = Rate(r-0.001);
					self.rate = vec![r-0.001;self.num_layers];
				},
				&mut Setting::WeightLimit(l) => {
					if self.weight_limit >= 2.0 { 
						*setting = WeightLimit(l-1.0);
						self.weight_limit = l - 1.0;
					}
				},
				&mut Setting::BatchSize(s) => {
					if self.batch_size >= 2 {
						*setting = BatchSize(s-1);
						self.batch_size= s - 1;
					}
				},
				&mut Setting::Datapoints(s) => {
					if self.datapoints >= 2 {
						*setting = Datapoints(s-1);
						self.datapoints= s - 1;
					}
				},
				&mut Setting::NodesInLayer{num_nodes :n , layer : l} => {
					if self.nodes_in_layer[l] >= 3 && l != 0 && l != self.num_layers - 1 {
						*setting = NodesInLayer{ num_nodes : n- 2 , layer :l };
						self.nodes_in_layer[l] -= 2;
						let next = self.nodes_in_layer.clone();
						self.makeover(&next);
					}
				},
				&mut Setting::ActivationOfLayer{act : f , layer : l} => {
					let next = match f {
						Identity => Sin,
						Relu => Identity,
						Tanh => Relu,
						Sin => Tanh,
					};
					*setting = ActivationOfLayer{act : next , layer : l};
					self.act[l] = next;
				},
				_ => (),
			}
		}
						
	
}
    