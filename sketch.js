let x_vals = [];    // im at 27.,28 in linear r0.2ion tf... shiffman
let y_vals = [];
let a, b, c, d;
let dragging = false;
firstdragComplete = false;
const learningRate = 0.4;
 const optimizer = tf.train.adam(learningRate);

function setup() {
  createCanvas(400, 400);
  
  a = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  c = tf.variable(tf.scalar(random(1))); // for parabolic
  d = tf.variable(tf.scalar(random(1))); // for cubic
}

function draw() {
  
  if (dragging) {
    let x = (map(mouseX, 0, width, -1, 1));
    let y = (map(mouseY, 0, height, 1, -1));
    x_vals.push(x);
    y_vals.push(y);
  } else {
    tf.tidy(() => {
      if(x_vals.length > 0) {
        const ys = tf.tensor1d(y_vals);
        optimizer.minimize(() => loss(predict(x_vals), ys));
      }
    });
  }
  
  background(0);
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    point(screen_pos_x(x_vals[i]), screen_pos_y(y_vals[i]));
  }
  
  console.log(tf.memory().numTensors);
  //noLoop();
  
  const curveX = [];
  for (let x = -1; x <= 1.0; x+=0.05) {
    curveX.push(x);
  } 
  
  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();
  
  beginShape();
  noFill();
    stroke(255);
    strokeWeight(2);
    for(let i = 0; i < curveX.length; i++) {
      let x = map(curveX[i], -1, 1, 0, width);
      let y = map(curveY[i], -1, 1, height, 0);
      vertex(x, y);
    }
  endShape();

}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = ax^2 + bx + c
  //const ys = xs.mul(m).add(b); // straight line
  //const ys = xs.square().mul(a).add(xs.mul(b)).add(c); //x^2
  const ys = xs.pow(tf.scalar(3)).mul(a).add(xs.square().mul(b)).add(xs.mul(c).add(d));  //x^3
  return ys;
}

function mousePressed() {
    // if (mouseButton === CENTER) {
    //   triangle(23, 75, 50, 20, 78, 75);
    // }
  dragging = true;
  firstdragComplete = false;
  // let x = mouseX;
  // let y = mouseY;
  // x = coord_pos_x(x);
  // y = coord_pos_y(y);
  // x_vals.push(x);
  // y_vals.push(y);
}

function mouseReleased() {
  dragging = false;
  firstdragComplete = true;
}

function keyPressed() {
    if (keyCode === DOWN_ARROW) {
      x_vals = [];
      y_vals = [];
      clear();
      background(0);
    }
    if (keyCode === UP_ARROW) {
      curveX = [];
      curveY = [];
      clear();
      background(0);
    }

}


function coord_pos_x(x) {
  x = map(x, 0, width, -1.0, 1.0);
  return x;
}

function coord_pos_y(y) {
  y = map(y, height, 0, -1.0, 1.0);
  return y;
}

function screen_pos_x(x) {
  x = map(x, -1.0, 1.0, 0, width);
  return x;
}

function screen_pos_y(y) {
  y = map(y, -1.0, 1.0, height, 0);
  return y;
}