
import random
import pandas as pd
import numpy as np

from scipy.stats import weibull_min


# 检验项	检验要求
PATH = './data/'

class data_generate():
    
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def mainteance_data_gen():

        samples = 100

        data = [[]]
        for i in range(0,samples) :
            # 发动机编号
            n1 = '发动机编号'
            v1 = 'engine_'+str(i)
            
            # 外观检查	有无缺陷
            n2 = '缸体外观'
            v2 = 1 
            # 缸体主轴孔同心度	≤0.075mm
            n3 = '缸体主轴孔同心度'
            v3 = random.uniform(0,0.075)
            
            # 缸体主轴孔内径	192.901mm≤I.D.≤192.951mm
            n4 = '缸体缸套上座孔内径'
            v4 = random.uniform(192.901,192.951)
            # 缸体缸套上座孔内径	190.28mm≤I.D.≤190.34mm
            n5 = '缸体缸套上座孔内径'
            v5 = random.uniform(190.28,190.34)
            # 缸体缸套下座孔内径	181.74mm≤I.D.≤181.80mm
            n6 = '缸体缸套下座孔内径'
            v6 = random.uniform(181.74,181.80)
            # 缸体缸套密封圈安装孔内径	177.32mm≤I.D.≤177.48mm
            n7 = '缸体缸套密封圈安装孔内径'
            v7 = random.uniform(177.32,177.48)
            # 缸套上座孔深度	13.684mm≤I.D.≤13.734mm
            n8 = '缸套上座孔深度'
            v8 = random.uniform(13.684,13.734)
            # 缸体惰轮轴孔内径	21.975mm≤I.D.≤22.025mm
            n9 = '缸体惰轮轴孔内径'
            v9 = random.uniform(21.975,22.025)
                
            # 外观检查	有无缺陷
            n10 = '曲轴外观'
            v10 = 1
            # 曲轴弯曲度	≤0.230mm
            n11 = '曲轴弯曲度'
            v11 = random.uniform(0,0.230)
            # 曲轴主轴颈	184.10mm≤O.D.≤184.15mm
            n12 = '曲轴主轴颈'
            v12 = random.uniform(184.10,184.15)
            # 曲轴连杆轴颈	120.57mm≤O.D.≤120.65mm
            n13 = '曲轴连杆轴颈'
            v13 = random.uniform(120.57,120.65)
            # 曲轴超声波探伤
            n14 = '曲轴超声波探伤'
            v14 = 0	
                
            # 外观检查	有无缺陷
            n15 = '凸轮轴外观'
            v15 = 1
            # 凸轮轴轴颈	104.987mm≤O.D.≤105.013mm
            n16 = '凸轮轴轴颈'
            v16 = random.uniform(104.987,105.013)
            # 凸轮轴轴颈跳动	≤0.08mm
            n17 = '凸轮轴轴颈跳动'
            v17 = random.uniform(0,0.08)
            # 凸轮轴超声波探伤	
            n18 = '凸轮轴超声波探伤'
            v18 = 0
                
            # 外观检查	有无缺陷

            # 水泵驱动轴	35.084mm≤E.C.≤35.096mm
            n19 = '水泵驱动轴'
            v19 = random.uniform(35.084,35.096)
            # 燃油泵驱动轴	34.964mm≤E.C.≤34.976mm
            n20 = '燃油泵驱动轴'
            v20 = random.uniform(34.964,34.976)
                
            # 外观检查	有无缺陷
            # 水泵惰轮衬套孔径	73.035mm≤E.C.≤73.065mm
            n21 = '水泵惰轮衬套孔径'
            v21 = random.uniform(73.035,73.065)
            # 液压泵惰轮衬套孔径
            n22 = '液压泵惰轮衬套孔径'
            v22 = random.uniform(0,1)
            # 右凸轮轴惰轮衬套孔径	
            n23 = '右凸轮轴惰轮衬套孔径'
            v23 = random.uniform(0,1)
            # 左凸轮轴惰轮衬套孔径	
            n24 = '左凸轮轴惰轮衬套孔径'
            v24 = random.uniform(0,1)
            # 水泵惰轮轴径	72.92mm≤E.C.≤72.98mm
            n25 = '水泵惰轮轴径'
            v25 = random.uniform(72.92,72.98)
            # 液压泵惰轮轴径
            n26 = '液压泵惰轮轴径'
            v26 = random.uniform(0,1)	
            # 右凸轮轴惰轴径
            n27 = '右凸轮轴惰轴径'
            v27 = random.uniform(0,1)
            # 左凸轮轴惰轴径
            n28 = '左凸轮轴惰轴径'
            v28 = random.uniform(0,1)	
                
            # 外观检查	有无缺陷
            n29 = '缸套外观'
            v29 = 1
            # 缸套内径椭圆度	≤0.064mm
            n30 = '缸套内径椭圆度'
            v30 = random.uniform(0,0.064)
            # 外观检查	有无缺陷
            # 活塞环开口间隙（上气环）（930e）	0.45≤I.D.≤0.65mm
            # 活塞环开口间隙（上气环）（5500）	0.56≤I.D.≤0.76mm
            n31 = '活塞环开口间隙上气环'
            v31 = random.uniform(0.45,0.65)
            # 活塞环开口间隙（下气环）	0.63≤I.D.≤1.02mm
            n32 = '活塞环开口间隙下气环'
            v32 = random.uniform(0.63,1.02)
            # 活塞环开口间隙（油环）	0.38≤I.D.≤0.76mm
            n33 = '活塞环开口间隙油环'
            v33 = random.uniform(0.38,0.76)
            # 外观检查	有无缺陷
            n34 = '连杆外观'
            v34 = 1
            # 连杆弯曲度	≤0.10mm
            n35 = '连杆弯曲度'
            v35 = random.uniform(0,0.10)
            # 连杆扭曲度	≤0.25mm
            n36 = '连杆扭曲度'
            v36 = random.uniform(0,0.25)
            # 连杆大头孔直径	127.032mm≤I.D.≤127.072mm
            n37 = '连杆大头孔直径'
            v37 = random.uniform(127.032,127.072)

            # 连杆活塞销衬套内径（旧）	65.025mm≤I.D.≤65.060mm
            # 连杆活塞销衬套内径（新）	65.035mm≤I.D.≤65.047mm
            n38 = '连杆活塞销衬套内径'
            v38 = random.uniform(65.035,65.060)
            # 选配连杆瓦	
                
            # 中冷器芯	≥0.5MPa，10min
            n39 = '中冷器芯'
            v39 = random.uniform(0.5,1)
            # 机油冷却器芯	≥0.5MPa，10min
            n40 = '机油冷却器芯'
            v40 = random.uniform(0.5,1)
            # 缸盖气水腔气密性	≥0.5MPa，10min
            n41 = '缸盖气水腔气密性'
            v41 = random.uniform(0.5,1)
                
            # 燃油泵供油量	≥
            n42 = '燃油泵供油量'
            v42 = random.uniform(0,1)
            # 喷油器雾化效果，开启压力	≤
            n43 = '喷油器雾化效果开启压力'
            v43 = random.uniform(0,1)
            # 喷油器供油量（多柱塞泵需测均匀性）	≤
            n44 = '喷油器供油量'
            v44 = random.uniform(0,1)
            # 喷油器雾化效果，开启压力	≥
            # 增压器叶轮轴向间隙，径向间隙，转动灵活性	≤
            n45 = '增压器叶轮轴向间隙'
            v45 = random.uniform(0,1)
            n46 = '增压器叶轮径向间隙'
            v46 = random.uniform(0,1)
            # 机油泵供油量，齿轮啮合间隙，轴向间隙，转动灵活性	≥
            n47 = '机油泵供油量'
            v47 = random.uniform(0,1)
            n48 = '机油泵齿轮啮合间隙'
            v48 = random.uniform(0,1)
            n49 = '机油泵轴向间隙'
            v49 = random.uniform(0,1)
            # 水流量，转动灵活性	≥
            n50 = '水泵流量'
            v50 = random.uniform(0,1)
            # 凸轮轴衬套内径	105.114≤I.D.≤105.177mm
            n51 = '凸轮轴衬套内径'
            v51 = random.uniform(105.114,105.177)

            # 专用工具固定螺栓Nm	68
            # 缸套压入(专用工具安装)	≤135Nm
            n52 = '缸套压入'
            v52 = random.uniform(0,135)
            # 曲轴回转力矩Nm	≤验证值
            n53 = '曲轴回转力矩'
            v53 = random.uniform(0,1)
            # 曲轴轴向间隙	0.13mm≤E.C.≤0.51mm
            n54 = '曲轴轴向间隙'
            v54 = random.uniform(0.13,0.51)
            # 连杆侧隙	0.30mm≤G≤0.51mm
            n55 = '连杆侧隙'
            v55 = random.uniform(0.30,0.51)
            # 主轴承盖螺栓Nm	195
            n56 = '主轴承盖螺栓'
            v56 = random.uniform(0,195)
            # 	420
            # 	回松
            # 	200
            # 	420
            # 	旋转 90 度
            # 主轴承盖侧向螺栓Nm	305
            n57 = '主轴承盖侧向螺栓'
            v57 = random.uniform(0,305)
            # 连杆螺栓Nm	305
            n58 = '连杆螺栓'
            v58 = random.uniform(0,305)
            # 	115
            # 	回松
            # 	95
            # 	旋转 60 度
            # 活塞冷却喷嘴安装螺栓M14 空心螺栓Nm	70
            n59 = '活塞冷却喷嘴安装螺栓M14'
            v59 = random.uniform(0,70)
            # 活塞冷却喷嘴安装螺栓M10 螺栓Nm	45
            n60 = '活塞冷却喷嘴安装螺栓M10'
            v60 = random.uniform(0,45)
            # 曲轴配重螺栓10.9 级螺钉Nm	240
            n61 = '曲轴配重螺栓10.9'
            v61 = random.uniform(0,240)
            # 曲轴配重螺栓12.9 级螺钉Nm	290
            n62 = '曲轴配重螺栓12.9'
            v62 = random.uniform(0,290)
                
            # 外观检查	有无缺陷
            # 缸套内径	158.737mm≤I.D.≤158.775mm
            n63 = '缸套内径'
            v63 = random.uniform(158.737,158.775)
            # 外观检查	有无缺陷
            # 凸轮轴轴向间隙	0.15mm≤E.C.≤0.33mm
            n64 = '凸轮轴轴向间隙'
            v64 = random.uniform(0.15,0.33)
            # 惰齿轴向间隙	0.15mm≤E.C.≤0.33mm
            n65 = '惰齿轴向间隙'
            v65 = random.uniform(0.15,0.33)
            # 齿轮齿隙	0.07mm≤T≤0.51mm
            n66 = '齿轮齿隙'
            v66 = random.uniform(0.07,0.51)
            # 发动机正时	0.334in
            n67 = '发动机正时'
            v67 = random.uniform(0,0.334)
            # 飞轮壳径向跳动930e/830e/4400/33900	≤0.25mm
            # 飞轮壳径向跳动5500	≤0.3mm
            n68 = '飞轮壳径向跳动'
            v68 = random.uniform(0,0.3)
            # 飞轮壳端面跳动930e/830e/4400/33900	≤0.25mm
            # 飞轮壳端面跳动5500	≤0.3mm
            n69 = '飞轮壳端面跳动'
            v69 = random.uniform(0,0.3)
            # 机油泵齿隙	0.21mm≤G≤0.39mm
            n70 = '机油泵齿隙'
            v70 = random.uniform(0.21,0.39)
            # 进气门间隙(过桥)	0.81mm（0.74mm≤T≤0.89mm）
            n71 = '进气门间隙过桥'
            v71 = random.uniform(0.74,0.89)
            # 排气门间隙(过桥)	0.36mm（0.36mm≤T≤0.51mm）
            n72 = '排气门间隙过桥'
            v72 = random.uniform(0.36,0.51)
            # 缸盖螺栓Nm	70
            # 	200
            # 	300
            # 	旋转 90 度
            # 前齿轮室螺栓Nm	80
            n73 = '前齿轮室螺栓'
            v73 = random.uniform(0,80)
            # 凸轮轴止推片安装螺栓Nm	45
            n74 = '凸轮轴止推片安装螺栓'
            v74 = random.uniform(0,45)
            # 惰轮固定螺栓Nm	80
            n75 = '惰轮固定螺栓'
            v75 = random.uniform(0,80)
            # 	165
            # 	280
            # 飞轮壳垫片螺栓Nm	9
            n76 = '飞轮壳垫片螺栓'
            v76 = random.uniform(0,9)

            # 飞轮壳螺栓Nm	95
            n77 = '飞轮壳螺栓'
            v77 = random.uniform(0,95)

            # 	195
            # 机油压力调节器塞堵Nm	60
            n78 = '机油压力调节器塞堵'
            v78 = random.uniform(0,60)
            # 后油封安装螺栓Nm	10
            n79 = '后油封安装螺栓'
            v79 = random.uniform(0,10)
            # 发动机吊架螺栓Nm	280
            n80 = '发动机吊架螺栓'
            v80 = random.uniform(0,280)
            # 检查孔盖Nm	45
            n81 = '检查孔盖'
            v81 = random.uniform(0,45)
            # 随动件Nm	280
            n82 = '随动件'
            v82 = random.uniform(0,280)
            # 油冷芯盖板安装螺栓Nm	45
            n83 = '油冷芯盖板安装螺栓'
            v83 = random.uniform(0,45)
            # 油冷芯盖板锁紧螺母Nm	95
            n84 = '油冷芯盖板锁紧螺母'
            v84 = random.uniform(0,95)
            # 前齿轮室盖螺栓M12 螺栓Nm	80
            n85 = '前齿轮室盖螺栓M12'
            v85 = random.uniform(0,80)
            # 前齿轮室盖螺栓M16 螺栓Nm	195
            n86 = '前齿轮室盖螺栓M16'
            v86 = random.uniform(0,195)
            # 机油泵安装M10螺栓Nm	45
            n87 = '机油泵安装M10螺栓'
            v87 = random.uniform(0,45)
            # 机油泵安装M12螺栓Nm	80
            n88 = '机油泵安装M12螺栓'
            v88 = random.uniform(0,80)
            # 油底壳所有安装螺栓Nm	45
            n89 = '油底壳所有安装螺栓'
            v89 = random.uniform(0,45)
            # 油底壳螺堵Nm	47
            n90 = '油底壳螺堵'
            v90 = random.uniform(0,47)
            # 节温器壳体Nm	80
            n91 = '节温器壳体'
            v91 = random.uniform(0,80)
                
            # 外观检查	有无缺陷
            # 燃烧室气密性	≤验证值
            # 气门锁夹正位	
                
            # 外观检查	有无缺陷
            # 燃油连接块M8螺栓Nm	23
            n92 = '燃油连接块M8螺栓'
            v92 = random.uniform(0,23)
            # 燃油连接块M12空心螺栓Nm	72
            n93 = '燃油连接块M12空心螺栓'
            v93 = random.uniform(0,72)
            # 油封及气门标记螺栓Nm	10
            n94 = '油封及气门标记螺栓'
            v94 = random.uniform(0,10)
            # 曲轴适配器螺栓Nm	200
            n95 = '曲轴适配器螺栓'
            v95 = random.uniform(0,200)
            # 	380
            # 	685
            # 曲轴减震器安装螺栓Nm	125
            n96 = '曲轴减震器安装螺栓'
            v96 = random.uniform(0,125)
            # 	165
            # 喷油器压板螺栓Nm	75
            n97 = '喷油器压板螺栓'
            v97 = random.uniform(0,75)
            # 摇臂螺栓Nm	280
            n98 = '摇臂螺栓'
            v98 = random.uniform(0,280)
            # 喷油器调整Nm（in/bl）	28（248）
            n99 = '喷油器调整'
            v99 = random.uniform(0,28)
            # 	回松
            # 	19（168）
            # 气门及喷油器调整螺丝锁紧螺母Nm	105
            n100 = '气门及喷油器调整螺丝锁紧螺母'
            v100 = random.uniform(0,105)
                
            # 外观检查	有无缺陷
            # 摇臂室安装螺栓Nm	115
            n101 = '摇臂室安装螺栓'
            v101 = random.uniform(0,115)

            # 摇臂室盖Nm	45
            n102 = '摇臂室盖'
            v102 = random.uniform(0,45)
            # 排气管螺栓Nm	45
            n103 = '排气管螺栓'
            v103 = random.uniform(0,45)
            # 排气管连接波纹管卡箍螺母Nm	10
            n104 = '排气管连接波纹管卡箍螺母'
            v104 = random.uniform(0,10)
            # 增压器安装螺母(法兰螺母)Nm	45
            n105 = '增压器安装螺母法兰螺母'
            v105 = random.uniform(0,45)
            # 增压器安装螺母(凸缘螺母)Nm	65
            n106 = '增压器安装螺母凸缘螺母'
            v106 = random.uniform(0,65)
            # 增压器回油管螺栓Nm	40
            n107 = '增压器回油管螺栓'
            v107 = random.uniform(0,40)
            # 低压增压器安装支架M10螺栓Nm	45
            n108 = '低压增压器安装支架M10螺栓'
            v108 = random.uniform(0,45)
            # 低压增压器安装支架M12螺栓Nm	80
            n109 = '低压增压器安装支架M12螺栓'
            v109 = random.uniform(0,80)
            # 低压增压器回油管安装螺栓Nm	45
            n110 = '低压增压器回油管安装螺栓'
            v110 = random.uniform(0,45)
                
            # 外观检查	有无缺陷
            # 马达安装螺栓Nm	215
            n111 = '马达安装螺栓'
            v111 = random.uniform(0,215)
            # 风扇轮毂Nm	280
            n112 = '风扇轮毂'
            v112 = random.uniform(0,280)
            # 中间冷器安装螺栓Nm	280
            n113 = '中间冷器安装螺栓'
            v113 = random.uniform(0,280)
            # 中间冷器进气胶管卡箍Nm	9
            n114 = '中间冷器进气胶管卡箍'
            v114 = random.uniform(0,9)
            # 中间冷器进回水法兰安装螺栓Nm	45
            n115 = '中间冷器进回水法兰安装螺栓'
            v115 = random.uniform(0,45)
            # 中间冷器进气歧管安装螺栓Nm	45
            n116 = '中间冷器进气歧管安装螺栓'
            v116 = random.uniform(0,45)
            # 燃油歧管安装螺栓Nm	9.5
            n117 = '燃油歧管安装螺栓'
            v117 = random.uniform(0,9.5)

            # 凸轮随动件盖安装螺栓Nm	45
            n118 = '凸轮随动件盖安装螺栓'
            v118 = random.uniform(0,45)
            # 中冷进回水法兰安装螺栓Nm	45
            n119 = '中冷进回水法兰安装螺栓'
            v119 = random.uniform(0,45)
            # 铝制中冷壳体总成安装螺栓Nm	45
            n120 = '铝制中冷壳体总成安装螺栓'
            v120 = random.uniform(0,45)
            # 铸铁中冷壳体总成安装螺栓Nm	65
            n121 = '铸铁中冷壳体总成安装螺栓'
            v121 = random.uniform(0,65)
            # 中冷器进气歧管安装Nm	45
            n122 = '中冷器进气歧管安装'
            v122 = random.uniform(0,45)
            # 中冷器进气胶管卡箍Nm	9
            n123 = '中冷器进气胶管卡箍'
            v123 = random.uniform(0,9)
            # 附件驱动安装螺栓Nm	45
            n124 = '附件驱动安装螺栓'
            v124 = random.uniform(0,45)
            # 燃油泵安装螺栓Nm	45
            n125 = '燃油泵安装螺栓'
            v125 = random.uniform(0,45)
            # 燃油泵支架安装螺栓Nm	80
            n126 = '燃油泵支架安装螺栓'
            v126 = random.uniform(0,80)
            # 水泵安装螺栓Nm	45
            n127 = '水泵安装螺栓'
            v127 = random.uniform(0,45)
            # 水泵支架螺栓Nm	45
            n128 = '水泵支架螺栓'
            v128 = random.uniform(0,45)
            # 水泵旁通管安装螺栓Nm	45
            n129 = '水泵旁通管安装螺栓'
            v129 = random.uniform(0,45)
            # LTA水泵安装螺栓Nm	80
            n130 = 'LTA水泵安装螺栓'
            v130 = random.uniform(0,80)
            # LTA水泵水管卡箍安装螺栓Nm	20
            n131 = 'LTA水泵水管卡箍安装螺栓'
            v131 = random.uniform(0,20)
            # 机油滤清器座安装螺栓Nm	45
            n132 = '机油滤清器座安装螺栓'
            v132 = random.uniform(0,45)
            # Eliminator™ 滤清器安装螺栓Nm	45
            n133 = 'Eliminator滤清器安装螺栓'
            v133 = random.uniform(0,45)
            # Eliminator™ 滤清器支架安装M10螺栓Nm	45
            n134 = 'Eliminator滤清器支架安装M10螺栓'
            v134 = random.uniform(0,45)
            # Eliminator™ 滤清器支架安装M12螺栓Nm	80
            n135 = 'Eliminator滤清器支架安装M12螺栓'
            v135 = random.uniform(0,80)
            # 呼吸器安装螺栓Nm	45
            n136 = '呼吸器安装螺栓'
            v136 = random.uniform(0,45)
            # ECVA安装螺栓Nm	45
            n137 = 'ECVA安装螺栓'
            v137 = random.uniform(0,45)
            # ECM安装螺栓Nm	9
            n138 = 'ECM安装螺栓'
            v138 = random.uniform(0,9)
            # 风扇皮带涨紧轮控制杆螺栓Nm	280
            n139 = '风扇皮带涨紧轮控制杆螺栓'
            v139 = random.uniform(0,280)
            # 燃油滤清器座安装螺栓Nm	60
            n140 = '燃油滤清器座安装螺栓'
            v140 = random.uniform(0,60)
            # LTA中冷水管P型夹固定螺栓Nm	45
            n141 = 'LTA中冷水管P型夹固定螺栓'
            v141 = random.uniform(0,45)
            
            data.append([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,
                        v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
                        v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,
                        v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,
                        v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,
                        v51,v52,v53,v54,v55,v56,v57,v58,v59,v60,
                        v61,v62,v63,v64,v65,v66,v67,v68,v69,v70,
                        v71,v72,v73,v74,v75,v76,v77,v78,v79,v80,
                        v81,v82,v83,v84,v85,v86,v87,v88,v89,v90,
                        v91,v92,v93,v94,v95,v96,v97,v98,v99,v100,
                        v101,v102,v103,v104,v105,v106,v107,v108,v109,v110,
                        v111,v112,v113,v114,v115,v116,v117,v118,v119,v120,
                        v121,v122,v123,v124,v125,v126,v127,v128,v129,v130,
                        v131,v132,v133,v134,v135,v136,v137,v138,v139,v140,
                        v141])
              
        columns = [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,
                n21,n22,n23,n24,n25,n26,n27,n28,n29,n30,
                n31,n32,n33,n34,n35,n36,n37,n38,n39,n40,
                n41,n42,n43,n44,n45,n46,n47,n48,n49,n50,
                n51,n52,n53,n54,n55,n56,n57,n58,n59,n60,
                n61,n62,n63,n64,n65,n66,n67,n68,n69,n70,
                n71,n72,n73,n74,n75,n76,n77,n78,n79,n80,
                n81,n82,n83,n84,n85,n86,n87,n88,n89,n90,
                n91,n92,n93,n94,n95,n96,n97,n98,n99,n100,
                n101,n102,n103,n104,n105,n106,n107,n108,n109,n110,
                n111,n112,n113,n114,n115,n116,n117,n118,n119,n120,
                n121,n122,n123,n124,n125,n126,n127,n128,n129,n130,
                n131,n132,n133,n134,n135,n136,n137,n138,n139,n140,
                n141]

        df = pd.DataFrame(columns = columns, data = data[1:])
        df.to_excel(f'{PATH}fakedata_engine_mainteance.xlsx', index=False)
        print('data generated')

    @staticmethod
    def water_pump_data_gen():
        
        total_samples =200
        positive_samples = 100
        
        data = [[]]
        for i in range(0,total_samples) :
            n1 = '编号'
            v1 = 'waterpump_'+str(i) 
            
            if i < positive_samples:
                n2 = '工况A流量'
                v2 = random.uniform(0.5,0.75)
                n3 = '工况B流量'
                v3 = random.uniform(0.75,0.1)           
                n4 = '质量判定'
                v4 = 1
                data.append([v1,v2,v3,v4])
            else:
                v2 = random.uniform(0.4,0.6)
                v3 = random.uniform(0.6,0.9)           
                v4 = 0
                data.append([v1,v2,v3,v4])
            
        columns = [n1,n2,n3,n4]
        df = pd.DataFrame(columns = columns, data = data[1:])
        df.to_excel(f'{PATH}fakedata_waterpump_mainteance.xlsx', index=False)
        print('data generated')

    # 生成数据。excel 100行 第一列低速流量，第二列高速流量，第三列检测时年龄，第四列最终寿命，
    # 其中检测时年龄小于最终寿命。寿命数值在300-1000之间，寿命分布符合weibull分布，
    # 低速流量和高速流量数据与检测时年龄数据非线性负相关，即年龄越大流量越小。
    
    @staticmethod
    def water_pump_lifedata_gen(num_rows=100):
        
        np.random.seed(42)
        # 生成最终寿命数据，符合 Weibull 分布，范围在 300-1000 之间
        shape, loc, scale = 1.5, 300, 700  # Weibull 分布的形状参数、位置参数和尺度参数
        lifetimes = weibull_min.rvs(shape, loc=loc, scale=scale, size=num_rows).astype(int)
        
        # 生成检测时年龄数据，确保年龄小于最终寿命
        ages = np.random.uniform(100, lifetimes - 1).astype(int)
        
        # 生成低速流量和高速流量数据，与检测时年龄非线性负相关
        base_low_flow = 50
        base_high_flow = 80
        low_flow_range = 20  # 范围 30-50之间的幅度
        high_flow_range = 20  # 范围 60-80之间的幅度

        # 计算非线性负相关的流量
        low_flow = base_low_flow - (low_flow_range / (1 + np.exp(-(ages - 50) / 10))) + np.random.normal(0, 1, num_rows)
        high_flow = base_high_flow - (high_flow_range / (1 + np.exp(-(ages - 50) / 10))) + np.random.normal(0, 1, num_rows)

        # # 确保流量在指定范围内
        # low_flow = np.clip(low_flow, 30, 50)
        # high_flow = np.clip(high_flow, 60, 80)

        df = pd.DataFrame({
            '低速流量': low_flow,
            '高速流量': high_flow,
            '检测时年龄': ages,
            '最终寿命': lifetimes
        })
        
        df.to_excel(f'{PATH}fakedata_waterpump_life.xlsx', index=False)
        print('data generated')

        
if __name__ == '__main__':
    data_generate.water_pump_lifedata_gen()
    
    
    # data_generate.mainteance_data_gen()
    # data_generate.water_pump_data_gen()
    