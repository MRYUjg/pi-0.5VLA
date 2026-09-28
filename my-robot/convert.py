import mujoco.viewer
 
def main():
    model = mujoco.MjModel.from_xml_path('E:/111PROject/robot-vla/openpi/my-robot/Grapper/urdf/crt_ctag2f90.urdf')
    output_xml_path = 'E:/111PROject/robot-vla/openpi/my-robot/Grapper/xml/zhixing_gripper.xml'
    mujoco.mj_saveLastXML(output_xml_path, model)
    print(f'Success! Saved to: {output_xml_path}')
 
if __name__ == '__main__':
    main()