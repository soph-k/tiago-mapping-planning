import py_trees                                         # Behavior tree library

from controller import Robot                            # Webots compatible robot handle
from config import RobotConfig                          # Global 
from utils import MemoryBoard, dev, enable, safe, rstep # Shared blackboard, device helpers
from navigation import NavigationController             # Mapping, planning, pathfollowing
from planning import PickPlaceController                # behavior tree
from camera import PerceptionController                 # Camera-based perception
from display import MapDisplay                          # Onscreen map trajectory display
from arms import BTAction                               # Adapter to use functions  in a behavior tree
from typing import Optional    

class RobotController:
    def __init__(self) -> None:
        self.robot = Robot()                            # Create simulator/hardware robot handle
        self.timestep = int(getattr(self.robot, "getBasicTimeStep", lambda: 32)())  # Control period 
        self.memory = MemoryBoard()                     # Shared key–value store across modules
        self.memory.set("robot", self.robot)            # Expose robot to others
        self.memory.set("timestep", self.timestep)      # Expose timestep to others
        self.navigation: Optional[NavigationController] = None           
        self.pickplace:  Optional[PickPlaceController]  = None           
        self.perception: Optional[PerceptionController] = None           
        self.display:    Optional[MapDisplay]           = None           
        self.state, self.step_counter, self.display_static, self.behavior_tree = "MAPPING", 0, False, None  # Init

    def _initialize_devices(self) -> None:
        # Pose sensors: GPS + Compass 
        gps, comp = dev(self.robot, 'gps'), dev(self.robot, 'compass')   # Lookup devices
        enable(gps, self.timestep); gps and self.memory.set("gps", gps)  # Start GPS; share if present
        enable(comp, self.timestep); comp and self.memory.set("compass", comp)  # Start compass; share if present
        # Differential drive motors 
        mL, mR = dev(self.robot, "wheel_left_joint"), dev(self.robot, "wheel_right_joint")  # Wheel motors
        safe(lambda: mL.setPosition(float('inf'))); safe(lambda: mR.setPosition(float('inf')))  # Velocity mode
        self.memory.set("motorL", mL); self.memory.set("motorR", mR)   # Share handles
        # Lidar for reactive avoidance and mapping ---
        lidar = None                                                   # Placeholder
        for name in ["Hokuyo URG-04LX-UG01", "lidar"]:                 # Try common device names
            l = dev(self.robot, name)                                  # Lookup device
            if l:
                enable(l, self.timestep)                               # Start scans
                lidar = l                                              # Use first found
                break
        lidar and self.memory.set("lidar", lidar)                      # Share lidar if found
        display = dev(self.robot, "display")                           # Lookup display
        display and self.memory.set("display", display)                # Share display handle
        cam = dev(self.robot, "camera")                                # Lookup camera
        enable(cam, self.timestep)                                     # Enable frames 
        cam and self.memory.set("camera", cam)                         # Share camera handle
        # Arm & head joints: collect motor handles and enable their position sensors
        arm_motors = {}                                                
        joint_names = [
            'torso_lift_joint','arm_1_joint','arm_2_joint','arm_3_joint',
            'arm_4_joint','arm_5_joint','arm_6_joint','arm_7_joint',
            'gripper_left_finger_joint','gripper_right_finger_joint',
            'head_1_joint','head_2_joint'
        ]                                                               # All joints we care about
        for j in joint_names:
            m = dev(self.robot, j)                                     # Find motor
            if m:
                arm_motors[j] = m                                      # Keep it
                safe(lambda m=m: m.getPositionSensor().enable(self.timestep))  # Enable joint sensor
        for s in ['gripper_left_sensor_finger_joint','gripper_right_sensor_finger_joint']:  # Fingertip sensors
            enable(dev(self.robot, s), self.timestep)                  # Best-effort enable
        self.memory.set("arm_motors", arm_motors)                      # Share arm motors

    def initialize(self) -> bool:
        try:
            print("Initializing robot devices...")
            self._initialize_devices()                                 # Probe/enable devices, store handles
            
            print("Building subsystem controllers...")
            # Construct subsystem controllers; share robot + memory 
            self.navigation = NavigationController(self.robot, self.memory)   # Mapping/planning
            self.pickplace  = PickPlaceController(self.robot, self.memory)    # Manipulation BT
            self.perception = PerceptionController(self.robot, self.memory)   # Camera recognition
            self.display    = MapDisplay(self.memory)                          # UI overlay/draw
            self.memory.set("navigation", self.navigation)                     # Let others find nav
            
            print("Loading saved maps...")
            # Try to load saved maps so we can skip live mapping if available
            c_loaded = self.navigation.load_maps(map_dir="map")         # Load on-disk cspace/prob map
            self.memory.set("cspace_loaded", c_loaded)                  # Remember whether loaded
            if c_loaded:
                print("Successfully loaded saved maps!")
            else:
                print("No saved maps found, will perform live mapping")
            
            print("Building behavior tree...")
            # Build a behavior tree: first mapping, then pick/place
            root = py_trees.composites.Sequence(name="RootSequence", memory=True)  # Sequence = do A then B
            root.add_children([BTAction(self._bt_mapping), BTAction(self._bt_pickplace)])  # Wrap functions as nodes
            self.behavior_tree = py_trees.trees.BehaviourTree(root)    # Create the tree
            safe(lambda: self.behavior_tree.setup(timeout=5))          # Initialize BT internals
            print("System initialization complete")
            return True                                                # Init OK
        except Exception as e:
            print(f"Initialization failed: {e}")                       # Log the error
            return False                                               # Signal failure

    def run(self) -> bool:
        if not self.initialize():                                       # Ensure subsystems created
            return False                                                # Abort if init failed
        print("Starting main control loop...")                          # Info line
        self.memory.set("system_instance", self)                        # Make controller discoverable

        try:
            while rstep(self.robot, self.timestep) != -1:               # Advance sim; -1 means done
                self.step_counter += 1                                  # Tick counter
                self.perception and self.perception.update()            # Lightweight perception pass
                if self.behavior_tree:                                  # Drive high-level behavior
                    self.behavior_tree.tick()                           # One BT tick
                    if self.behavior_tree.root.status == py_trees.common.Status.SUCCESS:
                        self.state = "COMPLETE"                         # Whole BT done
                if self.display and not self.display_static and self.step_counter % 15 == 0:
                    self.display.update()                               # UI update
                if self.step_counter > 20000:                           # Hard cap to avoid runaway loops
                    break                                               # Exit loop
                if self.state == "COMPLETE":                            # Mapping + pick/place finished
                    print("Task complete!")                             # Info
                    break                                               # Exit loop
        except KeyboardInterrupt:
            print("Interrupted")                             
        finally:
            print("Cleaning up and stopping robot...")
            self._cleanup()                                             # Try to stop the robot
        return True                                                     # Normal exit

    def _bt_mapping(self) -> str:
        if self.memory.get("cspace_loaded", False):                     # Saved map present?
            self.state = "MANIPULATION"                                 # Skip to next phase
            print("Mapping skipped - using saved map")                  # Info
            if self.display and not self.display_static:                # Draw once then freeze
                self.display.update()                                   # Render static map
                self.display_static = True                              # Stop live updates
            return "SUCCESS"                                            # Node succeeded
        if not self.navigation:                                         # Shouldn't happen if init succeeded
            return "FAILURE"                                            # Fail node
        r = self.navigation.execute_mapping()                           # Run mapping state machine
        if r == "SUCCESS":                                              # Finished mapping
            self.state = "MANIPULATION"                                 # Next phase
            print("Mapping completed")                                  # Info
            if self.display and not self.display_static:                # Draw once
                self.display.update()                                   # Render map
                self.display_static = True                              # Freeze updates
            return "SUCCESS"                                            # Node OK
        if r == "FAILURE":                                              # Mapping failed
            print("Mapping failed")                                     # Info
            return "FAILURE"                                            # Node failed
        return "RUNNING"                                                # Keep ticking

    def _bt_pickplace(self) -> str:
        if not self.pickplace:                                          # Defensive check
            return "FAILURE"                                            # Fail node
        r = self.pickplace.run()                                        # Tick manipulation tree
        if r == "SUCCESS":                                              # Success
            print("Pick and place completed")              # Info
            return "SUCCESS"                                            # Node OK
        if r == "FAILURE":                                              # Failure
            print("Pick and place failed")                              # Info
            return "FAILURE"                                            # Node failed
        return "RUNNING"                                                # Keep ticking

    def _cleanup(self) -> None:
        mL, mR = self.memory.get("motorL"), self.memory.get("motorR")   # Wheel handles
        try:
            mL and mL.setVelocity(0.0)                                   # Stop left wheel
            mR and mR.setVelocity(0.0)                                   # Stop right wheel
        except Exception:
            pass                                                         

    def test_pose(self, pose_name: str) -> bool:
        if not self.pickplace:                                          # Require manipulation stack
            return False                                                # Abort if missing
        from arms import ArmPoseController                               # Local import to avoid cycles
        ctrl = ArmPoseController(self.robot, self.memory, pose_name)     # Create pose helper
        ctrl.setup()                                                    # Resolve joints/sensors
        ctrl.initialise()                                               # Send targets
        while rstep(self.robot, self.timestep) != -1:                    # Step until pose reached
            if ctrl.update() == "SUCCESS":                              # Converged?
                break                                                   # Done
        return True                                                     # Reached pose


def main() -> None:
    print("Starting navigation and manipulation system")      # Entry log
    RobotController().run()                                             # Build, initialize, run
    print("System shutdown complete.")                                  # Exit log


if __name__ == "__main__":
    main()                                                              # Run when executed as a script
