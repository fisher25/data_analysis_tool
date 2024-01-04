
import random
# 检验项	检验要求
samples = 50

for i in range(0,samples) :
    # 发动机编号
    n1 = '发动机编号'
    engine_name = 'engine_'+str(i)
    
    # 外观检查	有无缺陷
    n2 = '缸体外观'
    cyl_outlook = 1 
    # 缸体主轴孔同心度	≤0.075mm
    n3 = '缸体主轴孔同心度'
    cyl_axle_cen = random.uniform(0,0.075)
    
    # 缸体主轴孔内径	192.901mm≤I.D.≤192.951mm
    n4 = '缸体缸套上座孔内径'
    cyl_axle_cen = random.uniform(192.901,192.951)
    # 缸体缸套上座孔内径	190.28mm≤I.D.≤190.34mm
    n5 = '缸体缸套上座孔内径'
    cyl_linerUp_hole_dia = random.uniform(190.28,190.34)
    # 缸体缸套下座孔内径	181.74mm≤I.D.≤181.80mm
    n6 = '缸体缸套下座孔内径'
    cyc_linerDown_hole_dia = random.uniform(181.74,181.80)
    # 缸体缸套密封圈安装孔内径	177.32mm≤I.D.≤177.48mm
    n7 = '缸体缸套密封圈安装孔内径'
    cyc_liner_seal_dia = random.uniform(177.32,177.48)
    # 缸套上座孔深度	13.684mm≤I.D.≤13.734mm
    n8 = '缸套上座孔深度'
    liner_hole_depth = random.uniform(13.684,13.734)
    # 缸体惰轮轴孔内径	21.975mm≤I.D.≤22.025mm
    n9 = '缸体惰轮轴孔内径'
    cyc_idler_hole_dia = random.uniform(21.975,22.025)
        
    # 外观检查	有无缺陷
    n10 = '曲轴外观'
    crk_outlook = 1
    # 曲轴弯曲度	≤0.230mm
    n11 = '曲轴弯曲度'
    crk_bend = random.uniform(0,0.230)
    # 曲轴主轴颈	184.10mm≤O.D.≤184.15mm
    n12 = '曲轴主轴颈'
    crk_axle_neck = random.uniform(184.10,184.15)
    # 曲轴连杆轴颈	120.57mm≤O.D.≤120.65mm
    n13 = '曲轴连杆轴颈'
    crk_rod_neck = random.uniform(120.57,120.65)
    # 曲轴超声波探伤
    n14 = '曲轴超声波探伤'
    crk_inner_spot = 0	
        
    # 外观检查	有无缺陷
    n15 = '凸轮轴外观'
    cam_outlook = 1
    # 凸轮轴轴颈	104.987mm≤O.D.≤105.013mm
    n16 = '凸轮轴轴颈'
    cam_axle_neck = random.uniform(104.987,105.013)
    # 凸轮轴轴颈跳动	≤0.08mm
    n17 = '凸轮轴轴颈跳动'
    cam_axle_neck_jump = random.uniform(0,0.08)
    # 凸轮轴超声波探伤	
    n18 = '凸轮轴超声波探伤'
    cam_inner_spot = 0
        
    # 外观检查	有无缺陷

    # 水泵驱动轴	35.084mm≤E.C.≤35.096mm
    n19 = '水泵驱动轴'
    water_pump_axle = random.uniform(35.084,35.096)
    # 燃油泵驱动轴	34.964mm≤E.C.≤34.976mm
    n20 = '燃油泵驱动轴'
    oil_pump_axle = random.uniform(34.964,34.976)
        
    # 外观检查	有无缺陷
    # 水泵惰轮衬套孔径	73.035mm≤E.C.≤73.065mm
    n21 = '水泵惰轮衬套孔径'
    water_pump_idler_holedia = random.uniform(73.035,73.065)
    # 液压泵惰轮衬套孔径
    n22 = '液压泵惰轮衬套孔径'
    pump_idler_holedia = random.uniform(0,1)
    # 右凸轮轴惰轮衬套孔径	
    n23 = '右凸轮轴惰轮衬套孔径'
    right_cam_idler_holedia = random.uniform(0,1)
    # 左凸轮轴惰轮衬套孔径	
    n24 = '左凸轮轴惰轮衬套孔径'
    left_cam_idler_holedia = random.uniform(0,1)
    # 水泵惰轮轴径	72.92mm≤E.C.≤72.98mm
    n25 = '水泵惰轮轴径'
    water_pump_idler_axledia = random.uniform(72.92,72.98)
    # 液压泵惰轮轴径
    n26 = '液压泵惰轮轴径'
    pump_idler_holedia = random.uniform(0,1)	
    # 右凸轮轴惰轴径
    n27 = '右凸轮轴惰轴径'
    right_cam_idler_axledia = random.uniform(0,1)
    # 左凸轮轴惰轴径
    n28 = '左凸轮轴惰轴径'
    left_cam_idler_holedia = random.uniform(0,1)	
        
    # 外观检查	有无缺陷
    n29 = '缸套外观'
    liner_outlook = 1
    # 缸套内径椭圆度	≤0.064mm
    n30 = '缸套内径椭圆度'
    liner_inner_ellipse = random.uniform(0,0.064)
    # 外观检查	有无缺陷
    # 活塞环开口间隙（上气环）（930e）	0.45≤I.D.≤0.65mm
    # 活塞环开口间隙（上气环）（5500）	0.56≤I.D.≤0.76mm
    n31 = '活塞环开口间隙上气环'
    piston_ring_gap_up = random.uniform(0.45,0.65)
    # 活塞环开口间隙（下气环）	0.63≤I.D.≤1.02mm
    n32 = '活塞环开口间隙下气环'
    piston_ring_gap_down = random.uniform(0.63,1.02)
    # 活塞环开口间隙（油环）	0.38≤I.D.≤0.76mm
    n33 = '活塞环开口间隙油环'
    piston_oilring_gap = random.uniform(0.38,0.76)
    # 外观检查	有无缺陷
    n34 = '连杆外观'
    rod_outlook = 1
    # 连杆弯曲度	≤0.10mm
    n35 = '连杆弯曲度'
    rod_bend = random.uniform(0,0.10)
    # 连杆扭曲度	≤0.25mm
    n36 = '连杆扭曲度'
    rod_twist = random.uniform(0,0.25)
    # 连杆大头孔直径	127.032mm≤I.D.≤127.072mm
    n37 = '连杆大头孔直径'
    rod_holedia = random.uniform(127.032,127.072)

    # 连杆活塞销衬套内径（旧）	65.025mm≤I.D.≤65.060mm
    # 连杆活塞销衬套内径（新）	65.035mm≤I.D.≤65.047mm
    n38 = '连杆活塞销衬套内径'
    rod_pistonpin_holedia = random.uniform(65.035,65.060)
    # 选配连杆瓦	
        
    # 中冷器芯	≥0.5MPa，10min
    n39 = '中冷器芯'
    cooler_tight = random.uniform(0.5,1)
    # 机油冷却器芯	≥0.5MPa，10min
    n40 = '机油冷却器芯'
    oil_cooler_tight = random.uniform(0.5,1)
    # 缸盖气水腔气密性	≥0.5MPa，10min
    n41 = '缸盖气水腔气密性'
    liner_tight = random.uniform(0.5,1)
        
    # 燃油泵供油量	≥
    n42 = '燃油泵供油量'
    oil_pump_flow = random.uniform(0,1)
    # 喷油器雾化效果，开启压力	≤
    n43 = '喷油器雾化效果开启压力'
    injector_pressure = random.uniform(0,1)
    # 喷油器供油量（多柱塞泵需测均匀性）	≤
    n44 = '喷油器供油量'
    injector_flow = random.uniform(0,1)
    # 喷油器雾化效果，开启压力	≥
    # 增压器叶轮轴向间隙，径向间隙，转动灵活性	≤
    n45 = '增压器叶轮轴向间隙'
    compressor_axle_gap = random.uniform(0,1)
    n46 = '增压器叶轮径向间隙'
    compressor_radial_gap = random.uniform(0,1)
    # 机油泵供油量，齿轮啮合间隙，轴向间隙，转动灵活性	≥
    n47 = '机油泵供油量'
    lube_pump_flow = random.uniform(0,1)
    n48 = '机油泵齿轮啮合间隙'
    lube_pump_gear_gap = random.uniform(0,1)
    n49 = '机油泵轴向间隙'
    lube_pump_axle_gap = random.uniform(0,1)
    # 水流量，转动灵活性	≥
    n50 = '水泵流量'
    water_pump_flow = random.uniform(0,1)
    # 凸轮轴衬套内径	105.114≤I.D.≤105.177mm
    n51 = '凸轮轴衬套内径'
    cam_bushing_dia = random.uniform(105.114,105.177)

    # 专用工具固定螺栓Nm	68
    # 缸套压入(专用工具安装)	≤135Nm
    n52 = '缸套压入'
    liner_install = random.uniform(0,135)
    # 曲轴回转力矩Nm	≤验证值
    n53 = '曲轴回转力矩'
    crk_rotate_torque = random.uniform(0,1)
    # 曲轴轴向间隙	0.13mm≤E.C.≤0.51mm
    n54 = '曲轴轴向间隙'
    crk_axle_gap = random.uniform(0.13,0.51)
    # 连杆侧隙	0.30mm≤G≤0.51mm
    n55 = '连杆侧隙'
    rod_side_gap = random.uniform(0.30,0.51)
    # 主轴承盖螺栓Nm	195
    n56 = '主轴承盖螺栓'
    axle_bolt = random.uniform(0,195)
    # 	420
    # 	回松
    # 	200
    # 	420
    # 	旋转 90 度
    # 主轴承盖侧向螺栓Nm	305
    n57 = '主轴承盖侧向螺栓'
    axle_side_bolt = random.uniform(0,305)
    # 连杆螺栓Nm	305
    n58 = '连杆螺栓'
    rod_bolt = random.uniform(0,305)
    # 	115
    # 	回松
    # 	95
    # 	旋转 60 度
    # 活塞冷却喷嘴安装螺栓M14 空心螺栓Nm	70
    n59 = '活塞冷却喷嘴安装螺栓M14'
    piston_cooling_bolt_M14 = random.uniform(0,70)
    # 活塞冷却喷嘴安装螺栓M10 螺栓Nm	45
    n60 = '活塞冷却喷嘴安装螺栓M10'
    piston_cooling_bolt_M10 = random.uniform(0,45)
    # 曲轴配重螺栓10.9 级螺钉Nm	240
    n61 = '曲轴配重螺栓10.9'
    crk_bolt_10_9 = random.uniform(0,240)
    # 曲轴配重螺栓12.9 级螺钉Nm	290
    n62 = '曲轴配重螺栓12.9'
    crk_bolt_12_9 = random.uniform(0,290)
        
    # 外观检查	有无缺陷
    # 缸套内径	158.737mm≤I.D.≤158.775mm
    n63 = '缸套内径'
    liner_inner_dia = random.uniform(158.737,158.775)
    # 外观检查	有无缺陷
    # 凸轮轴轴向间隙	0.15mm≤E.C.≤0.33mm
    n64 = '凸轮轴轴向间隙'
    cam_axle_gap = random.uniform(0.15,0.33)
    # 惰齿轴向间隙	0.15mm≤E.C.≤0.33mm
    n65 = '惰齿轴向间隙'
    idler_axle_gap = random.uniform(0.15,0.33)
    # 齿轮齿隙	0.07mm≤T≤0.51mm
    n66 = '齿轮齿隙'
    idler_gear_gap = random.uniform(0.07,0.51)
    # 发动机正时	0.334in
    n67 = '发动机正时'
    engine_timing = random.uniform(0,0.334)
    # 飞轮壳径向跳动930e/830e/4400/33900	≤0.25mm
    # 飞轮壳径向跳动5500	≤0.3mm
    n68 = '飞轮壳径向跳动'
    flywheel_radial_jump = random.uniform(0,0.3)
    # 飞轮壳端面跳动930e/830e/4400/33900	≤0.25mm
    # 飞轮壳端面跳动5500	≤0.3mm
    n69 = '飞轮壳端面跳动'
    flywheel_verical_jump = random.uniform(0,0.3)
    # 机油泵齿隙	0.21mm≤G≤0.39mm
    n70 = '机油泵齿隙'
    lube_pump_gear_gap = random.uniform(0.21,0.39)
    # 进气门间隙(过桥)	0.81mm（0.74mm≤T≤0.89mm）
    n71 = '进气门间隙过桥'
    intake_valve_gap = random.uniform(0.74,0.89)
    # 排气门间隙(过桥)	0.36mm（0.36mm≤T≤0.51mm）
    n72 = '排气门间隙过桥'
    outtake_valve_gap = random.uniform(0.36,0.51)
    # 缸盖螺栓Nm	70
    # 	200
    # 	300
    # 	旋转 90 度
    # 前齿轮室螺栓Nm	80
    n73 = '前齿轮室螺栓'
    gear_room_bolt = random.uniform(0,80)
    # 凸轮轴止推片安装螺栓Nm	45
    n74 = '凸轮轴止推片安装螺栓'
    cam_stop_bolt = random.uniform(0,45)
    # 惰轮固定螺栓Nm	80
    n75 = '惰轮固定螺栓'
    idler_bolt = random.uniform(0,80)
    # 	165
    # 	280
    # 飞轮壳垫片螺栓Nm	9
    n76 = '飞轮壳垫片螺栓'
    flywheel_shim_bolt = random.uniform(0,9)

    # 飞轮壳螺栓Nm	95
    n77 = '飞轮壳螺栓'
    flywheel_bolt = random.uniform(0,95)

    # 	195
    # 机油压力调节器塞堵Nm	60
    n78 = '机油压力调节器塞堵'
    lube_oil_pressure_bolt = random.uniform(0,60)
    # 后油封安装螺栓Nm	10
    n79 = '后油封安装螺栓'
    back_oil_seal_bolt = random.uniform(0,10)
    # 发动机吊架螺栓Nm	280
    n80 = '发动机吊架螺栓'
    engine_hang_bolt = random.uniform(0,280)
    # 检查孔盖Nm	45
    n81 = '检查孔盖'
    check_hole_bolt = random.uniform(0,45)
    # 随动件Nm	280
    n82 = '随动件'
    follower_bolt = random.uniform(0,280)
    # 油冷芯盖板安装螺栓Nm	45
    n83 = '油冷芯盖板安装螺栓'
    oil_cooler_bolt = random.uniform(0,45)
    # 油冷芯盖板锁紧螺母Nm	95
    n84 = '油冷芯盖板锁紧螺母'
    oil_cooler_nut = random.uniform(0,95)
    # 前齿轮室盖螺栓M12 螺栓Nm	80
    n85 = '前齿轮室盖螺栓M12'
    front_gear_room_bolt_M12 = random.uniform(0,80)
    # 前齿轮室盖螺栓M16 螺栓Nm	195
    n86 = '前齿轮室盖螺栓M16'
    front_gear_room_bolt_M16 = random.uniform(0,195)
    # 机油泵安装M10螺栓Nm	45
    n87 = '机油泵安装M10螺栓'
    lube_pump_bolt_M10 = random.uniform(0,45)
    # 机油泵安装M12螺栓Nm	80
    n88 = '机油泵安装M12螺栓'
    lube_pump_bolt_M12 = random.uniform(0,80)
    # 油底壳所有安装螺栓Nm	45
    n89 = '油底壳所有安装螺栓'
    lube_bottom_bolt = random.uniform(0,45)
    # 油底壳螺堵Nm	47
    n90 = '油底壳螺堵'
    lube_bottom_plug = random.uniform(0,47)
    # 节温器壳体Nm	80
    n91 = '节温器壳体'
    temperature_shell = random.uniform(0,80)
        
    # 外观检查	有无缺陷
    # 燃烧室气密性	≤验证值
    # 气门锁夹正位	
        
    # 外观检查	有无缺陷
    # 燃油连接块M8螺栓Nm	23
    n92 = '燃油连接块M8螺栓'
    oil_connection_bolt_M8 = random.uniform(0,23)
    # 燃油连接块M12空心螺栓Nm	72
    n93 = '燃油连接块M12空心螺栓'
    oil_connection_bolt_M8 = random.uniform(0,72)
    # 油封及气门标记螺栓Nm	10
    n94 = '油封及气门标记螺栓'
    back_oil_seal_bolt = random.uniform(0,10)
    # 曲轴适配器螺栓Nm	200
    n95 = '曲轴适配器螺栓'
    crk_adapter_bolt = random.uniform(0,200)
    # 	380
    # 	685
    # 曲轴减震器安装螺栓Nm	125
    n96 = '曲轴减震器安装螺栓'
    crk_vibration_bolt = random.uniform(0,125)
    # 	165
    # 喷油器压板螺栓Nm	75
    n97 = '喷油器压板螺栓'
    injector_plate_bolt = random.uniform(0,75)
    # 摇臂螺栓Nm	280
    n98 = '摇臂螺栓'
    swing_arm_bolt = random.uniform(0,280)
    # 喷油器调整Nm（in/bl）	28（248）
    n99 = '喷油器调整'
    injector_adjust = random.uniform(0,28)
    # 	回松
    # 	19（168）
    # 气门及喷油器调整螺丝锁紧螺母Nm	105
    n100 = '气门及喷油器调整螺丝锁紧螺母'
    injector_adjust_nut = random.uniform(0,105)
        
    # 外观检查	有无缺陷
    # 摇臂室安装螺栓Nm	115
    n101 = '摇臂室安装螺栓'
    swing_arm_bolt = random.uniform(0,115)

    # 摇臂室盖Nm	45
    n102 = '摇臂室盖'
    swing_arm_cover_bolt = random.uniform(0,45)
    # 排气管螺栓Nm	45
    n103 = '排气管螺栓'
    exhaust_pipe_bolt = random.uniform(0,45)
    # 排气管连接波纹管卡箍螺母Nm	10
    n104 = '排气管连接波纹管卡箍螺母'
    exhaust_pipe_nut = random.uniform(0,10)
    # 增压器安装螺母(法兰螺母)Nm	45
    n105 = '增压器安装螺母法兰螺母'
    compressor__flange_nut = random.uniform(0,45)
    # 增压器安装螺母(凸缘螺母)Nm	65
    n106 = '增压器安装螺母凸缘螺母'
    compressor_nut = random.uniform(0,65)
    # 增压器回油管螺栓Nm	40
    n107 = '增压器回油管螺栓'
    compressor_return_bolt = random.uniform(0,40)
    # 低压增压器安装支架M10螺栓Nm	45
    n108 = '低压增压器安装支架M10螺栓'
    low_compressor_bolt_M10 = random.uniform(0,45)
    # 低压增压器安装支架M12螺栓Nm	80
    n109 = '低压增压器安装支架M12螺栓'
    low_compressor_bolt_M12 = random.uniform(0,80)
    # 低压增压器回油管安装螺栓Nm	45
    n110 = '低压增压器回油管安装螺栓'
    low_compressor_return_bolt = random.uniform(0,45)
        
    # 外观检查	有无缺陷
    # 马达安装螺栓Nm	215
    n111 = '马达安装螺栓'
    motor_bolt = random.uniform(0,215)
    # 风扇轮毂Nm	280
    n112 = '风扇轮毂'
    fan_hub_bolt = random.uniform(0,280)
    # 中间冷器安装螺栓Nm	280
    n113 = '中间冷器安装螺栓'
    middle_cooler_bolt = random.uniform(0,280)
    # 中间冷器进气胶管卡箍Nm	9
    n114 = '中间冷器进气胶管卡箍'
    middle_cooler_air_pipe_nut = random.uniform(0,9)
    # 中间冷器进回水法兰安装螺栓Nm	45
    n115 = '中间冷器进回水法兰安装螺栓'
    middle_cooler_flange_bolt = random.uniform(0,45)
    # 中间冷器进气歧管安装螺栓Nm	45
    n116 = '中间冷器进气歧管安装螺栓'
    middle_cooler_air_manifold_bolt = random.uniform(0,45)
    # 燃油歧管安装螺栓Nm	9.5
    n117 = '燃油歧管安装螺栓'
    oil_pipe_bolt = random.uniform(0,9.5)

    # 凸轮随动件盖安装螺栓Nm	45
    n118 = '凸轮随动件盖安装螺栓'
    cam_follower_cover_bolt = random.uniform(0,45)
    # 中冷进回水法兰安装螺栓Nm	45
    n119 = '中冷进回水法兰安装螺栓'
    middle_cooler_return_flange_bolt = random.uniform(0,45)
    # 铝制中冷壳体总成安装螺栓Nm	45
    n120 = '铝制中冷壳体总成安装螺栓'
    middle_cooler_aluminum_bolt = random.uniform(0,45)
    # 铸铁中冷壳体总成安装螺栓Nm	65
    n121 = '铸铁中冷壳体总成安装螺栓'
    middle_cooler_iron_bolt = random.uniform(0,65)
    # 中冷器进气歧管安装Nm	45
    n122 = '中冷器进气歧管安装'
    middle_cooler_pipe_bolt = random.uniform(0,45)
    # 中冷器进气胶管卡箍Nm	9
    n123 = '中冷器进气胶管卡箍'
    middle_cooler_air_pipe_nut = random.uniform(0,9)
    # 附件驱动安装螺栓Nm	45
    n124 = '附件驱动安装螺栓'
    accessory_bolt = random.uniform(0,45)
    # 燃油泵安装螺栓Nm	45
    n125 = '燃油泵安装螺栓'
    oil_pump_bolt = random.uniform(0,45)
    # 燃油泵支架安装螺栓Nm	80
    n126 = '燃油泵支架安装螺栓'
    oil_pump_support_bolt = random.uniform(0,80)
    # 水泵安装螺栓Nm	45
    n127 = '水泵安装螺栓'
    water_pump_bolt = random.uniform(0,45)
    # 水泵支架螺栓Nm	45
    n128 = '水泵支架螺栓'
    water_pump_support_bolt = random.uniform(0,45)
    # 水泵旁通管安装螺栓Nm	45
    n129 = '水泵旁通管安装螺栓'
    water_pump_pipe_bolt = random.uniform(0,45)
    # LTA水泵安装螺栓Nm	80
    n130 = 'LTA水泵安装螺栓'
    LTA_pump_bolt = random.uniform(0,80)
    # LTA水泵水管卡箍安装螺栓Nm	20
    n131 = 'LTA水泵水管卡箍安装螺栓'
    LTA_pump_pipe_bolt = random.uniform(0,20)
    # 机油滤清器座安装螺栓Nm	45
    n132 = '机油滤清器座安装螺栓'
    lube_filter_bolt = random.uniform(0,45)
    # Eliminator™ 滤清器安装螺栓Nm	45
    n133 = 'Eliminator滤清器安装螺栓'
    Eli_lube_filter_bolt = random.uniform(0,45)
    # Eliminator™ 滤清器支架安装M10螺栓Nm	45
    n134 = 'Eliminator滤清器支架安装M10螺栓'
    Eli_lube_filter_bolt_M10 = random.uniform(0,45)
    # Eliminator™ 滤清器支架安装M12螺栓Nm	80
    n135 = 'Eliminator滤清器支架安装M12螺栓'
    Eli_lube_filter_bolt_M12 = random.uniform(0,80)
    # 呼吸器安装螺栓Nm	45
    n136 = '呼吸器安装螺栓'
    breather_bolt = random.uniform(0,45)
    # ECVA安装螺栓Nm	45
    n137 = 'ECVA安装螺栓'
    ECVA_bolt = random.uniform(0,45)
    # ECM安装螺栓Nm	9
    n138 = 'ECM安装螺栓'
    ECM_bolt = random.uniform(0,9)
    # 风扇皮带涨紧轮控制杆螺栓Nm	280
    n139 = '风扇皮带涨紧轮控制杆螺栓'
    fan_belt_bolt = random.uniform(0,280)
    # 燃油滤清器座安装螺栓Nm	60
    n140 = '燃油滤清器座安装螺栓'
    oil_filter_bolt = random.uniform(0,60)
    # LTA中冷水管P型夹固定螺栓Nm	45
    n141 = 'LTA中冷水管P型夹固定螺栓'
    LTA_middle_cooler_bolt = random.uniform(0,45)


# new_row = []
# df.loc[len(df)] = new_row