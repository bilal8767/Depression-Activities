const FILENAME = "eatingmi.csv";
const SAMPLING_INTERVAL = 1000;

Bangle.setCompassPower(true);


let latestMag = { x: 0, y: 0, z: 0 };
Bangle.on('mag', mag => latestMag = mag);

let file = require("Storage").open(FILENAME, "w");
file.write("time,ax,ay,az,mx,my,mz\n"); 

let interval = setInterval(() => {
  const time = Date.now();
  const accel = Bangle.getAccel();
  const mag = latestMag;

  const line = [
    time,
    accel.x.toFixed(3),
    accel.y.toFixed(3),
    accel.z.toFixed(3),
    mag.x.toFixed(3),
    mag.y.toFixed(3),
    mag.z.toFixed(3)
  ].join(",") + "\n";

  file.write(line);
  console.log("Logged:", line.trim());
}, SAMPLING_INTERVAL);

setWatch(() => {
  clearInterval(interval);
  Bangle.setCompassPower(false); 
  file = null; // closes the file
  console.log("Logging stopped.");
  Bangle.buzz();
}, BTN1, { repeat: false, edge: "rising" });

console.log("Logging started. Press BTN1 to stop.");
Bangle.setLCDPower(1);