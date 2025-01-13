#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create an instance of the PCA9685 PWM driver
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Define the number of angles in each array
const int numAngles = 50;

// Define arrays to hold the angles for each servo
float angles1[] = {
-112.146030,
-113.878407,
-115.363216,
-116.650108,
-117.769854,
-118.742796,
-119.583052,
-120.300821,
-120.903752,
-121.397806,
-121.787823,
-122.077910,
-122.271715,
-122.372618,
-122.383869,
-122.308678,
-122.150275,
-121.911944,
-121.597019,
-121.208881,
-120.750919,
-120.226489,
-119.638861,
-118.991152,
-118.286268,
-117.526836,
-116.715141,
-115.853071,
-114.942061,
-113.983051,
-112.976440,
-111.922056,
-110.819111,
-109.666167,
-108.461082,
-107.200944,
-105.881969,
-104.499362,
-103.047104,
-101.517632,
-99.901355, 
-98.185894, 
-96.354864, 
-94.385805, 
-92.246464, 
-89.887491, 
-87.226190, 
-84.102743, 
-80.114004, 
-72.640867, 
-72.640867, 
-75.263307, 
-77.127722, 
-78.699336, 
-80.100884, 
-81.386979, 
-82.587632, 
-83.721443, 
-84.800885, 
-85.834797, 
-86.829709, 
-87.790603, 
-88.721372, 
-89.625127, 
-90.504390, 
-91.361238, 
-92.197397, 
-93.014320, 
-93.813234, 
-94.595187, 
-95.361076, 
-96.111671, 
-96.847638, 
-97.569551, 
-98.277907, 
-98.973136, 
-99.655607, 
-100.325638,
-100.983503,
-101.629431,
-102.263617,
-102.886221,
-103.497372,
-104.097173,
-104.685699,
-105.263002,
-105.829109,
-106.384028,
-106.927744,
-107.460225,
-107.981418,
-108.491250,
-108.989630,
-109.476451,
-109.951582,
-110.414878,
-110.866169,
-111.305268,
-111.731966,
-112.146030
};

 int angles2[]={39.122463,
 44.032360,
 48.436842,
 52.446890,
 56.131417,
 59.535910,
 62.691665,
 65.620837,
 68.339406,
 70.859037,
 73.188286,
 75.333431,
 77.299051,
 79.088448,
 80.703964,
 82.147217,
 83.419287,
 84.520862,
 85.452351,
 86.213972,
 86.805821,
 87.227920,
 87.480255,
 87.562792,
 87.475481,
 87.218253,
 86.790991,
 86.193504,
 85.425474,
 84.486395,
 83.375498,
 82.091653,
 80.633252,
 78.998065,
 77.183050,
 75.184125,
 72.995858,
 70.611055,
 68.020204,
 65.210674,
62.165566,
58.861956,
55.268134,
51.338975,
47.007674,
42.169557,
36.646110,
30.088118,
21.607722,
5.408529,
5.408529,
10.211593,
13.329768,
15.794037,
17.876371,
19.697802,
21.324861,
22.798808,
24.147331,
25.390081,
26.541599,
27.612999,
28.612996,
29.548571,
30.425410,
31.248216,
32.020922,
32.746856,
33.428858,
34.069368,
34.670499,
35.234090,
35.761747,
36.254882,
36.714737,
37.142409,
37.538867,
 37.904967,
 38.241470,
 38.549044,
 38.828282,
 39.079702,
 39.303761,
 39.500851,
 39.671312,
 39.815433,
 39.933451,
 40.025561,
 40.091910,
 40.132606,
 40.147712,
 40.137254,
 40.101215,
 40.039536,
 39.952121,
 39.838826,
 39.699468,
 39.533815,
 39.341590,
 39.122463};

// Define a variable to keep track of the current index in the arrays
int currentIndex = 0;

// Define min and max pulse length out of 4096 (for the PCA9685)
#define SERVOMIN 150 // Minimum pulse length count (approx. 0 degrees)
#define SERVOMAX 600 // Maximum pulse length count (approx. 180 degrees)

void setup() {
  Serial.begin(9600);

  // Initialize the PCA9685
  pwm.begin();
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz

  // Optional: Give some time to ensure servos reach initial position
  delay(1000);
}

void loop() {
  // Forward motion through the angles
 /* for (currentIndex = 0; currentIndex < numAngles; currentIndex++) {
    moveServos();
    delay(50);
  }
*/
  // Retrace motion to initial position
 /* for (currentIndex = numAngles - 1; currentIndex >= 0; currentIndex--) {
    moveServos();
    delay(50);
  }*/
  moveServos();
    //delay(50);
    currentIndex++;
  if(currentIndex==numAngles-1)
  currentIndex=0;

  // Optional: Pause before restarting the loop
}

void moveServos() {
  // Read the angles from the arrays for the current index
  int i=0,j=0,b_angle=0,flag=0,l=10,r=10;
  for(i=0,j=50;i<50,j<100;i++,j++)//left leg lift 
  {
  float angle1 = angles1[i];
  float angle2 = angles2[i];
  float angle3 = angles1[j];
  float angle4 = angles2[j];


  // Calculate the pulse width for each angle
  int pulse1 = map(180+angle1-l, 0, 180, SERVOMIN, SERVOMAX);
  int pulse2 = map(90+angle2, 0, 180, SERVOMIN, SERVOMAX);
  int pulse3 = map(180+angle3-r, 0, 180, SERVOMIN, SERVOMAX);
  int pulse4 = map(90-angle4, 0, 180, SERVOMIN, SERVOMAX);
  int pulse6 = map(90-b_angle, 0, 180, SERVOMIN, SERVOMAX);
  

  // Move the servos to the desired angles
  pwm.setPWM(0, 0, pulse1); // Channel 0 for servo1
  pwm.setPWM(1, 0, pulse2);
  pwm.setPWM(2, 0, pulse3);
  pwm.setPWM(3, 0, pulse4);
  pwm.setPWM(5, 0, pulse6); // Channel 0 for servo1
  if(b_angle<=25 && flag==0)
  b_angle=b_angle+1;
  else
  {flag=1;
  b_angle=b_angle-1;
  }
  Serial.println(b_angle);
  delay(50);
  }
  b_angle=0; // Channel 1 for2
  int pulse6 = map(90, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(5, 0, pulse6);
  flag=0;
   for(i=50,j=0;i<100,j<50;i++,j++)
  {
  float angle1 = angles1[i];
  float angle2 = angles2[i];
  float angle3 = angles1[j];
  float angle4 = angles2[j];

  // Calculate the pulse width for each angle
  int pulse1 = map(180+angle1-l, 0, 180, SERVOMIN, SERVOMAX);
  int pulse2 = map(90+angle2, 0, 180, SERVOMIN, SERVOMAX);
  int pulse3 = map(180+angle3-r, 0, 180, SERVOMIN, SERVOMAX);
  int pulse4 = map(90-angle4, 0, 180, SERVOMIN, SERVOMAX);
  int pulse5 = map(90+b_angle, 0, 180, SERVOMIN, SERVOMAX);//left leg bottom servo


  // Move the servos to the desired angles
  pwm.setPWM(0, 0, pulse1); // Channel 0 for servo1
  pwm.setPWM(1, 0, pulse2);
  pwm.setPWM(2, 0, pulse3);
  pwm.setPWM(3, 0, pulse4);
  pwm.setPWM(4, 0, pulse5); // Channel 0 for servo1
  if(b_angle<=25 && flag==0)
  b_angle=b_angle+1;
  else
  {flag=1;
  b_angle=b_angle-1;
  }
  //Serial.println(b_angle);
  delay(50);
  }
  int pulse5 = map(90, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(4, 0, pulse5); // Channel 1 for servo2

  // Output the angles and index for debugging
 /* Serial.print(angle1);
  Serial.print(" ");
  Serial.print(angle2);
  Serial.print(" ");
  Serial.print(currentIndex);
  Serial.print(" ");
  Serial.println("");*/
}
