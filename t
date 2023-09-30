SiriusBad:
一、常用代码

地图全开(复制下面两行，中间要回车)：

minimap = TheSim:FindFirstEntityWithTag("minimap")

minimap.MiniMap:ShowArea(0,0,0, 10000)

全物品解锁(不消耗材料)：

GetP ().components.builder:GiveAllRecipes()

精神值回复满：c_setsanity(1)

恢复血量：c_give("healingsalve",99)

获得食物：c_give("turkeydinner",99)

二、代码大全

1、材料类

c_give("cutgrass",99)(草)

c_give("twigs",99)(树枝)

c_give("log",99)(木头)

c_give("charcoal",99)(木炭)

c_give("ash",99)(灰)

c_give("cutreeds",99)(采下的芦苇)

c_give("lightbulb",99)(荧光果)

c_give("petals",99)(花瓣)

c_give("petals_evil",99)(噩梦花瓣)

c_give("pinecone",99)(松果)

c_give("foliage",99)(叶子)

c_give("cutlichen",99)(摘下的苔藓)

c_give("wormlight",99)(虫子果)

c_give("lureplantbulb",99)(食人花种子)

c_give("flint",99)(燧石)

c_give("nitre",99)(硝石)

c_give("redgem",99)(红宝石)

c_give("bluegem",99)(蓝宝石)

c_give("purplegem",99)(紫宝石)

c_give("greengem",99)(绿宝石)

c_give("orangegem",99)(橙宝石)

c_give("yellowgem",99)(黄宝石)

c_give("rocks",99)(岩石)

c_give("goldnugget",99)(黄金)

c_give("thulecite",99)(铥矿石)

c_give("thulecite_pieces",99)(铥矿碎片)

c_give("rope",99)(绳子)

c_give("boards",99)(木板)

c_give("cutstone",99)(石砖)

c_give("papyrus",99)(纸)

c_give("houndstooth",99)(犬牙)

c_give("pigskin",99)(猪皮)

c_give("manrabbit_tail",99)(兔人尾巴)

c_give("silk",99)(蜘蛛丝)

c_give("spidergland",99)(蜘蛛腺体)

c_give("spidereggsack",99)(蜘蛛卵)

c_give("beardhair",99)(胡子)

c_give("beefalowool",99)(牛毛)

c_give("honeycomb",99)(蜂巢)

c_give("stinger",99)(蜂刺)

c_give("walrus_tusk",99)(海象牙)

c_give("feather_crow",99)(乌鸦羽毛)

c_give("feather_robin",99)(红雀羽毛)

c_give("feather_robin_winter",99)(雪雀羽毛)

c_give("horn",99)(牛角)

c_give("tentaclespots",99)(触手皮)

c_give("trunk_summer",99)(夏象鼻)

c_give("trunk_winter",99)(冬象鼻)

c_give("slurtleslime",99)(蜗牛龟粘液)

c_give("slurtle_shellpieces",99)(蜗牛龟壳片)

c_give("butterflywings",99)(蝴蝶翅膀)

c_give("mosquitosack",99)(蚊子血囊)

c_give("slurper_pelt",99)(啜食者皮)

c_give("minotaurhorn",99)(远古守护者角)

c_give("deerclops_eyeball",99)(巨鹿眼球)

c_give("lightninggoathorn",99)(闪电羊角)

c_give("glommerwings",99)(格罗门翅膀)

c_give("glommerflower",99)(格罗门花)

c_give("glommerfuel",99)(格罗门燃料)

c_give("livinglog",99)(活木头)

c_give("nightmarefuel",99)(噩梦燃料)

c_give("gears",99)(齿轮)

c_give("transistor",99)(晶体管)

c_give("marble",99)(大理石)

c_give("boneshard",99)(硬骨头)

c_give("ice",99)(冰)

c_give("poop",99)(便便)

c_give("guano",99)(鸟粪)

c_give("dragon_scales",99)(蜻蜓鳞片)

c_give("goose_feather",99)(鹿鸭羽毛)

c_give("coontail",99)(浣熊尾巴)

c_give("bearger_fur,99)(熊皮)

c_give("monstermeat",99)(怪物肉)

SiriusBad:
2、工具武器类

DebugSpawn"axe"(斧子)

DebugSpawn"goldenaxe"(黄金斧头)

DebugSpawn"lucy"(露西斧子)

DebugSpawn"hammer"(锤子)

DebugSpawn"pickaxe"(镐)

DebugSpawn"goldenpickaxe"(黄金镐)

DebugSpawn"shovel"(铲子)

DebugSpawn"goldenshovel"(黄金铲子)

DebugSpawn"pitchfork"(草叉)

DebugSpawn"razor"(剃刀)

DebugSpawn"bugnet"(捕虫网)

DebugSpawn"fishingrod"(鱼竿)

DebugSpawn"multitool_axe_pickaxe"(多功能工具)

DebugSpawn"cane"(行走手杖)

DebugSpawn"trap"(陷阱)

DebugSpawn"birdtrap"(鸟陷阱)

DebugSpawn"trap_teeth"(牙齿陷阱)

DebugSpawn"trap_teeth_maxwell"(麦斯威尔的牙齿陷阱

DebugSpawn"backpack"(背包)

DebugSpawn"piggyback"(猪皮包)

DebugSpawn"krampus_sack"(坎普斯背包)

DebugSpawn"umbrella"(雨伞)

DebugSpawn"grass_umbrella"(草伞)

DebugSpawn"heatrock"(保温石)

DebugSpawn"bedroll_straw"(草席卷)

DebugSpawn"bedroll_furry"(毛皮铺盖

DebugSpawn"torch"(火炬)

DebugSpawn"lantern"(提灯)

DebugSpawn"pumpkin_lantern"(南瓜灯)

DebugSpawn"compass"(指南针)

DebugSpawn"fertilizer"(化肥)

DebugSpawn"firesuppressor"(灭火器

DebugSpawn"sewing_kit"(缝纫工具包

DebugSpawn"spear"(矛)

DebugSpawn"boomerang"(回旋镖)

DebugSpawn"tentaclespike"(狼牙棒)

DebugSpawn"blowdart_pipe"(吹箭)

DebugSpawn"blowdart_sleep"(麻醉吹箭)

DebugSpawn"blowdart_fire"(燃烧吹箭

DebugSpawn"hambat"(火腿短棍)

DebugSpawn"nightsword"(暗影剑)

DebugSpawn"batbat"(蝙蝠棒)

DebugSpawn"ruins_bat"(远古短棒)

DebugSpawn"spear_wathgrithr"(瓦丝格雷斯矛)

DebugSpawn"panflute"(排箫)

DebugSpawn"onemanband"(独奏乐器)

DebugSpawn"gunpowder"(火药)

DebugSpawn"beemine"(蜜蜂地雷)

DebugSpawn"bell"(铃)

DebugSpawn"amulet"(红色护身符)

DebugSpawn"blueamulet"(蓝色护身符

DebugSpawn"purpleamulet"(紫色护身符)

DebugSpawn"yellowamulet"(黄色护身符)

DebugSpawn"orangeamulet"(橙色护身符)

DebugSpawn"greenamulet"(绿色护身符)

DebugSpawn"nightmare_timepiece"(铥矿奖章)

DebugSpawn"icestaff"(冰魔杖)

DebugSpawn"firestaff"(火魔杖)

DebugSpawn"telestaff"(传送魔杖)

DebugSpawn"orangestaff"(橙色魔杖)

DebugSpawn"greenstaff"(绿色魔杖)

DebugSpawn"yellowstaff"(黄色魔杖)

DebugSpawn"diviningrod"(探矿杖)

DebugSpawn"book_birds"(召唤鸟的书)

DebugSpawn"book_tentacles"(召唤触手的书)

DebugSpawn"book_gardening"(催生植物的书)

DebugSpawn"book_sleep"(催眠的书)

DebugSpawn"book_brimstone"(召唤闪电的书)

DebugSpawn"waxwelljournal"(麦斯威尔的日志)

DebugSpawn·(阿比盖尔之花)

DebugSpawn"balloons_empty"(空气球)

DebugSpawn"balloon"(气球)

DebugSpawn"lighter"(薇洛的打火机)

DebugSpawn"chester_eyebone"(切斯特骨眼)

DebugSpawn"featherfan"(羽毛扇)

DebugSpawn"staff_tornado"(龙卷风魔杖)

DebugSpawn"nightstick"(夜棍)

DebugSpawn"Coconade"(椰弹)

SiriusBad:
3、穿戴类

DebugSpawn"strawhat"(草帽)

DebugSpawn"flowerhat"(花环)

DebugSpawn"beefalohat"(牛毛帽)

DebugSpawn"featherhat"(羽毛帽)

DebugSpawn"footballhat"(猪皮帽)

DebugSpawn"tophat"(高礼帽)

DebugSpawn"earmuffshat"(兔耳罩)

DebugSpawn"winterhat"(冬帽)

DebugSpawn"minerhat"(矿工帽)

DebugSpawn"spiderhat"(蜘蛛帽)

DebugSpawn"beehat"(蜂帽)

DebugSpawn"walrushat"(海象帽)

DebugSpawn"slurtlehat"(蜗牛帽子)

DebugSpawn"bushhat"(丛林帽)

DebugSpawn"ruinshat"(远古王冠)

DebugSpawn"rainhat"(防雨帽)

DebugSpawn"icehat"(冰帽)

DebugSpawn"watermelonhat"(西瓜帽)

DebugSpawn"catcoonhat"(浣熊帽)

DebugSpawn"wathgrithrhat"(瓦丝格雷斯帽)

DebugSpawn"armorwood"(木盔甲)

DebugSpawn"armorgrass"(草盔甲)

DebugSpawn"armormarble"(大理石盔甲)

DebugSpawn"armor_sanity"(夜魔盔甲)

DebugSpawn"armorsnurtleshell"(蜗牛龟盔甲)

DebugSpawn"armorruins"(远古盔甲)

DebugSpawn"sweatervest"(小巧背心)

DebugSpawn"trunkvest_summer"(夏日背心)

DebugSpawn"trunkvest_winter"(寒冬背心)

DebugSpawn"armorslurper"(饥饿腰带)

DebugSpawn"raincoat"(雨衣)

DebugSpawn"webberskull"(韦伯头骨)

DebugSpawn"molehat"(鼹鼠帽)

DebugSpawn"armordragonfly"(蜻蜓盔甲)

DebugSpawn"beargervest"(熊背心)

DebugSpawn"eyebrellahat"(眼睛帽)

DebugSpawn"reflectivevest"(反射背心)

DebugSpawn"hawaiianshirt"(夏威夷衬衫)

4、建筑类

DebugSpawn"campfire"(营火)

DebugSpawn"firepit"(石头营火)

DebugSpawn"coldfire"(冷火)

DebugSpawn"coldfirepit"(石头冷火)

DebugSpawn"cookpot"(锅)

DebugSpawn"icebox"(冰箱)

DebugSpawn"winterometer"(寒冰温度计)

DebugSpawn"rainometer"(雨量计)

DebugSpawn"slow_farmplot"(一般农田)

DebugSpawn"fast_farmplot"(高级农田)

DebugSpawn"siestahut"(午睡小屋)

DebugSpawn"tent"(帐篷)

DebugSpawn"homesign"(路牌)

DebugSpawn"birdcage"(鸟笼)

DebugSpawn"meatrack"(晾肉架)

DebugSpawn"lightning_rod"(避雷针)

DebugSpawn"pottedfern"(盆栽)

DebugSpawn"nightlight"(暗夜照明灯)

DebugSpawn"nightmarelight"(影灯)

DebugSpawn"researchlab"(科学机器)

DebugSpawn"researchlab2"(炼金术引擎)

DebugSpawn"researchlab3"(阴影操纵者)

DebugSpawn"researchlab4"(灵子分解器)

DebugSpawn"treasurechest"(木箱)

DebugSpawn"skullchest"(骷髅箱)

DebugSpawn"pandoraschest"(华丽的箱子)

DebugSpawn"minotaurchest"(大华丽的箱子)

DebugSpawn"dragonflychest"(蜻蜓箱子)

DebugSpawn"wall_hay_item"(草墙)

DebugSpawn"wall_wood_item"(木墙)

DebugSpawn"wall_stone_item"(石墙)

DebugSpawn"wall_ruins_item"(铥墙)

DebugSpawn"wall_hay"(地上的草墙)

DebugSpawn"wall_wood"(地上的木墙)

DebugSpawn"wall_stone"(地上的石墙)

DebugSpawn"wall_ruins"(地上的铥墙)

DebugSpawn"pighouse"(猪房)

DebugSpawn"rabbithole"(兔房)

DebugSpawn"mermhouse"(鱼人房)

DebugSpawn"resurrectionstatue"(肉块雕像)

DebugSpawn"resurrectionstone"(重生石)

DebugSpawn"ancient_altar" (远古祭坛)

DebugSpawn"ancient_altar_broken "(损坏的远古祭坛)

DebugSpawn"tele "(传送核心)

DebugSpawn"gemsocket"(宝石看台)

DebugSpawn"eyeturret"(固定在地上的眼睛炮塔)

DebugSpawn"eyeturret_item"(可带走的眼睛炮塔)

DebugSpawn"cave_exit"(洞穴出口)

DebugSpawn"turf_woodfloor"(木地板)

DebugSpawn"turf_carpetfloor"(地毯地板)

DebugSpawn"turf_checkerfloor"(棋盘地板)

DebugSpawn"adventure_portal"(冒险之门)

DebugSpawn"rock_light"(火山坑)

DebugSpawn"gravestone"(墓碑)

DebugSpawn"mound"(坟墓土堆)

DebugSpawn"skeleton"(人骨)

DebugSpawn"houndbone"(狗骨头)

DebugSpawn"animal_track"(动物足迹)

DebugSpawn"dirtpile"(可疑的土堆)

DebugSpawn"pond"(池塘)

DebugSpawn"pond_cave"(洞穴池塘)

DebugSpawn"pighead"(猪头棍)

DebugSpawn"mermhead"(鱼头棍)

DebugSpawn"pigtorch"(猪火炬)

DebugSpawn"rabbithole"(兔子洞)

DebugSpawn"beebox"(蜂箱)

DebugSpawn"beehive"(野生蜂窝)

DebugSpawn"wasphive"(杀人蜂窝)

DebugSpawn"spiderhole"(洞穴蜘蛛洞)

DebugSpawn"walrus_camp"(海象窝)

DebugSpawn"tallbirdnest"(高鸟窝)

DebugSpawn"houndmound"(猎犬丘)

DebugSpawn"slurtlehole"(蜗牛窝)

DebugSpawn"batcave"(蝙蝠洞)

DebugSpawn"monkeybarrel"(猴子桶)

DebugSpawn"spiderden"(蜘蛛巢穴)

DebugSpawn"molehill"(鼹鼠丘)

DebugSpawn"catcoonden"(浣熊洞)

DebugSpawn"rock1"(带硝石的岩石)

DebugSpawn"rock2"(带黄金的岩石)

DebugSpawn"rock_flintless"(只有石头的岩石)

DebugSpawn"stalagmite_full"(大圆洞穴石头)

DebugSpawn"stalagmite_med"(中圆洞穴石头)

DebugSpawn"stalagmite_low"(小圆洞穴石头)

DebugSpawn"stalagmite_tall_full"(大高洞穴石头)

DebugSpawn"stalagmite_tall_med"(中高洞穴石头)

DebugSpawn"stalagmite_tall_low"(小高洞穴石头)

DebugSpawn"rock_ice"(冰石)

DebugSpawn"ruins_statue_head"(远古头像)

DebugSpawn"ruins_statue_mage"(远古法师雕像)

DebugSpawn"marblepillar"(大理石柱子)

DebugSpawn"marbletree"(大理石树)

DebugSpawn"statueharp"(竖琴雕像)

DebugSpawn"basalt"(玄武岩)

DebugSpawn"basalt_pillar"(高玄武岩)

DebugSpawn"insanityrock"(猪王矮柱石)

DebugSpawn"sanityrock"(猪王高柱石)

DebugSpawn"ruins_chair"(远古椅子)

DebugSpawn"ruins_vase"(远古花瓶)

DebugSpawn"ruins_table"(远古桌子)

DebugSpawn"statuemaxwell"(麦斯威尔雕像)

DebugSpawn"statueglommer"(格罗门雕像)

DebugSpawn"relic"(废墟)

DebugSpawn"ruins_rubble"(损毁的废墟)

DebugSpawn"bishop_nightmare"(损坏的雕像)

DebugSpawn"rook_nightmare"(损坏的战车)

DebugSpawn"knight_nightmare"(损坏的骑士)

DebugSpawn"chessjunk1"(损坏的机械1)

DebugSpawn"chessjunk2"(损坏的机械2)

DebugSpawn"chessjunk3"(损坏的机械3)

DebugSpawn"teleportato_ring"(环状传送机零件)

DebugSpawn"teleportato_box"(盒状传送机零件)

DebugSpawn"teleportato_crank"(曲柄状传送机零件)

DebugSpawn"teleportato_potato"(球状传送机零件)

DebugSpawn"teleportato_ "(传送机零件底座)

DebugSpawn"teleportato_checkmate"(传送机零件底座)

DebugSpawn"wormhole"(虫洞)

DebugSpawn"wormhole_limited_1"(被限制的虫洞)

DebugSpawn"stafflight"(小星星)

DebugSpawn"treasurechest_trap"(箱子陷阱)

DebugSpawn"icepack"(冰包)

DebugSpawn"Supertelescope"(超级望远镜)

5、食物类

DebugSpawn"carrot"(胡萝卜)

DebugSpawn"carrot_cooked"(熟胡萝卜)

DebugSpawn"berries"(浆果)

DebugSpawn"berries_cooked"(熟浆果)

DebugSpawn"pumpkin"(南瓜)

DebugSpawn"pumpkin_cooked"(熟南瓜)

DebugSpawn"dragonfruit"(火龙果)

DebugSpawn"dragonfruit_cooked"(熟火龙果)

DebugSpawn"pomegranate"(石榴)

DebugSpawn"pomegranate_cooked"(熟石榴)

DebugSpawn"corn"(玉米)

DebugSpawn"corn_cooked"(熟玉米)

DebugSpawn"durian"(榴莲)

DebugSpawn"durian_cooked"(熟榴莲)

DebugSpawn"eggplant"(茄子)

DebugSpawn"eggplant_cooked"(熟茄子)

DebugSpawn"cave_banana"(洞穴香蕉)

DebugSpawn"cave_banana_cooked"(熟洞穴香蕉)

DebugSpawn"acorn"(橡果)

DebugSpawn"acorn_cooked"(熟橡果)

DebugSpawn"cactus_meat"(仙人掌肉)

DebugSpawn"watermelon"(西瓜)

DebugSpawn"red_cap"(采摘的红蘑菇)

DebugSpawn"red_cap_cooked"(煮熟的红蘑菇)

DebugSpawn"green_cap"(采摘的绿蘑菇)

DebugSpawn"green_cap_cooked"(煮熟的绿蘑菇)

DebugSpawn"blue_cap_cooked"(煮熟的蓝蘑菇)

DebugSpawn"blue_cap"(采摘的蓝蘑菇)

DebugSpawn"seeds"(种子)

DebugSpawn"seeds_cooked"(熟种子)

DebugSpawn"carrot_seeds"(胡萝卜种子)

DebugSpawn"pumpkin_seeds"(南瓜种子)

DebugSpawn"dragonfruit_seeds"(火龙果种子)

DebugSpawn"pomegranate_seeds"(石榴种子)

DebugSpawn"corn_seeds"(玉米种子)

DebugSpawn"durian_seeds"(榴莲种子)

DebugSpawn"eggplant_seeds"(茄子种子)

DebugSpawn"smallmeat"(小肉)

DebugSpawn"cookedsmallmeat"(小熟肉)

DebugSpawn"smallmeat_dried"(小干肉)

DebugSpawn"meat"(大肉)

DebugSpawn"cookedmeat"(大熟肉)

DebugSpawn"meat_dried"(大干肉)

DebugSpawn"drumstick"(鸡腿)

DebugSpawn"drumstick_cooked"(熟鸡腿)

DebugSpawn"monstermeat"(疯肉)

DebugSpawn"cookedmonstermeat"(熟疯肉)

DebugSpawn"monstermeat_dried"(干疯肉)

DebugSpawn"plantmeat"(食人花肉)

DebugSpawn"plantmeat_cooked"(熟食人花肉)

DebugSpawn"bird_egg"(鸡蛋)

DebugSpawn"bird_egg_cooked"(煮熟的鸡蛋)

DebugSpawn"rottenegg"(烂鸡蛋)

DebugSpawn"tallbirdegg"(高鸟蛋)

DebugSpawn"tallbirdegg_cooked"(熟高鸟蛋)

DebugSpawn"tallbirdegg_cracked"(孵化的高鸟蛋)

DebugSpawn"fish"(鱼)

DebugSpawn"fish_cooked"(熟鱼)

DebugSpawn"eel"(鳗鱼)

DebugSpawn"eel_cooked"(熟鳗鱼)

DebugSpawn"froglegs"(蛙腿)

DebugSpawn"froglegs_cooked"(熟蛙腿)

DebugSpawn"batwing"(蝙蝠翅膀)

DebugSpawn"batwing_cooked"(熟蝙蝠翅膀)

DebugSpawn"trunk_cooked"(熟象鼻)

DebugSpawn"mandrake"(曼德拉草)

DebugSpawn"cookedmandrake"(熟曼特拉草)

DebugSpawn"honey"(蜂蜜)

DebugSpawn"butter"(黄油)

DebugSpawn"butterflymuffin"(奶油松饼)

DebugSpawn"frogglebunwich"(青蛙圆面包三明治)

DebugSpawn"honeyham"(蜜汁火腿)

DebugSpawn"dragonpie"(龙馅饼)

DebugSpawn"taffy"(太妃糖)

DebugSpawn"pumpkincookie"(南瓜饼)

DebugSpawn"kabobs"(肉串)

DebugSpawn"powcake"(芝士蛋糕)

DebugSpawn"mandrakesoup"(曼德拉草汤)

DebugSpawn"baconeggs"(鸡蛋火腿)

DebugSpawn"bonestew"(肉汤)

DebugSpawn"perogies"(半圆小酥饼)

DebugSpawn"wetgoop"(湿腻焦糊)

DebugSpawn"ratatouille"(蹩脚的炖菜)

DebugSpawn"fruitmedley"(水果拼盘)

DebugSpawn"fishtacos" (玉米饼包炸鱼)

DebugSpawn"waffles" (华夫饼)

DebugSpawn"turkeydinner"(火鸡正餐)

DebugSpawn"fishsticks"(鱼肉条)

DebugSpawn"stuffedeggplant"(香酥茄盒)

DebugSpawn"honeynuggets"(甜蜜金砖)

DebugSpawn"meatballs"(肉丸)

DebugSpawn"jammypreserves"(果酱蜜饯)

DebugSpawn"monsterlasagna"(怪物千层饼)

DebugSpawn"unagi"(鳗鱼料理)

DebugSpawn"bandage"(蜂蜜绷带)

DebugSpawn"healingsalve"(治疗药膏)

DebugSpawn"spoiled_food"(腐烂食物)

DebugSpawn"flowersalad"(花沙拉)

DebugSpawn"icecream"(冰激淋)

DebugSpawn"watermelonicle"(西瓜冰)

DebugSpawn"trailmix"(干果)

DebugSpawn"hotchili"(咖喱)

DebugSpawn"guacamole"(鳄梨酱)

DebugSpawn"goatmilk"(羊奶)

6、植物类

DebugSpawn"flower"(花)

DebugSpawn"flower_evil"(噩梦花)

DebugSpawn"carrot_planted"(长在地上的胡萝卜)

DebugSpawn"grass"(长在地上的草)

DebugSpawn"depleted_grass"(草根)

DebugSpawn"dug_grass"(长草簇)

DebugSpawn"sapling"(树苗)

DebugSpawn"dug_sapling"(可种的树苗)

DebugSpawn"berrybush"(果树丛)

DebugSpawn"dug_berrybush"(可种的果树丛)

DebugSpawn"berrybush2"(果树丛2)

DebugSpawn"dug_berrybush2"(可种的果树丛2)

DebugSpawn"marsh_bush"(尖刺灌木)

DebugSpawn"dug_marsh_bush"(可种的尖刺灌木)

DebugSpawn"reeds"(芦苇)

DebugSpawn"lichen"(洞穴苔藓)

DebugSpawn"cave_fern"(蕨类植物)

DebugSpawn"evergreen"(树)

DebugSpawn"evergreen_sparse"(无松果的树)

DebugSpawn"marsh_tree"(针叶树)

DebugSpawn"cave_banana_tree"(洞穴香蕉树)

DebugSpawn"livingtree"(活树)

DebugSpawn"deciduoustree"(橡树)

DebugSpawn"deciduoustree_tall"(高橡树)

DebugSpawn"deciduoustree_short"(矮橡树)

DebugSpawn"red_mushroom"(红蘑菇)

DebugSpawn"green_mushroom"(绿蘑菇)

DebugSpawn"blue_mushroom"(蓝蘑菇)

DebugSpawn"mushtree_tall"(高蘑菇树)

DebugSpawn"mushtree_medium"(中蘑菇树)

DebugSpawn"mushtree_small"(小蘑菇树)

DebugSpawn"flower_cave"(单朵洞穴花)

DebugSpawn"flower_cave_double"(双朵洞穴花)

DebugSpawn"flower_cave_triple"(三朵洞穴花)

DebugSpawn"tumbleweed"(滚草)

DebugSpawn"cactus"(仙人掌)

DebugSpawn"cactus_flower"(仙人掌花)

DebugSpawn"marsh_plant"(水塘边小草)

DebugSpawn"pond_algae"(水藻)

7、动物类

DebugSpawn"rabbit"(兔子)

DebugSpawn"perd"(火鸡)

DebugSpawn"crow"(乌鸦)

DebugSpawn"robin"(红雀)

DebugSpawn"robin_winter"(雪雀)

DebugSpawn"butterfly"(蝴蝶)

DebugSpawn"fireflies"(萤火虫)

DebugSpawn"bee"(蜜蜂)

DebugSpawn"killerbee"(杀人蜂)

DebugSpawn"flies"(苍蝇)

DebugSpawn"mosquito"(蚊子)

DebugSpawn"frog"(青蛙)

DebugSpawn"beefalo"(牛)

DebugSpawn"babybeefalo"(小牛)

DebugSpawn"lightninggoat"(闪电羊)

DebugSpawn"pigman"(猪人)

DebugSpawn"pigguard"(猪守卫)

DebugSpawn"bunnyman"(兔人)

DebugSpawn"merm"(鱼人)

DebugSpawn"spider_hider"(洞穴蜘蛛)

DebugSpawn"spider_spitter"(喷射蜘蛛)

DebugSpawn"spider"(地面小蜘蛛)

DebugSpawn"spider_warrior"(地面绿蜘蛛)

DebugSpawn"spiderqueen"(蜘蛛女王)

DebugSpawn"spider_dropper"(白蜘蛛)

DebugSpawn"hound"(猎狗)

DebugSpawn"firehound"(红色猎狗)

DebugSpawn"icehound"(冰狗)

DebugSpawn"tentacle"(触手)

DebugSpawn"tentacle_garden"(巨型触手)

DebugSpawn"leif"(树精)

DebugSpawn"leif_sparse"(稀有树精)

DebugSpawn"walrus"(海象)

DebugSpawn"little_walrus"(小海象)

DebugSpawn"smallbird"(小高鸟)

DebugSpawn"teenbird"(青年高鸟)

DebugSpawn"tallbird"(高鸟)

DebugSpawn"koalefant_summer"(夏象)

DebugSpawn"koalefant_winDter"(冬象)

DebugSpawn"penguin"(企鹅)

DebugSpawn"slurtle"(蜗牛龟)

DebugSpawn"snurtle"(黏糊虫)

DebugSpawn"bat"(蝙蝠)

DebugSpawn"rocky"(龙虾)

DebugSpawn"monkey"(猴子)

DebugSpawn"slurper"(缀食者)

DebugSpawn"buzzard"(秃鹫)

DebugSpawn"mole"(鼹鼠)

DebugSpawn"catcoon"(浣熊)

DebugSpawn"knight"(发条骑士)

DebugSpawn"bishop"(主教)

DebugSpawn"rook"(战车)

DebugSpawn"crawlinghorror"(爬行暗影怪)

DebugSpawn"terrorbeak"(尖嘴暗影怪)

DebugSpawn"deerclops"(巨鹿)

DebugSpawn"minotaur"(远古守护者)

DebugSpawn"worm"(远古虫子)

DebugSpawn"abigail"(阿比盖尔)

DebugSpawn"ghost"(幽灵)

DebugSpawn"shadowwaxwell"(麦斯威尔黑影小人

DebugSpawn"krampus"(坎普斯)

DebugSpawn"glommer"(格罗门)

DebugSpawn"chester"(切斯特)

DebugSpawn"lureplant"(食人花)

DebugSpawn"eyeplant"(食人花眼睛)

DebugSpawn"bigfoot"(大脚)

DebugSpawn"pigking"(猪王)

DebugSpawn"moose"(鹿鸭)

DebugSpawn"mossling"(小鸭)

DebugSpawn"dragonfly"(蜻蜓)

DebugSpawn"warg"(座狼)

DebugSpawn"bearger"(熊)

DebugSpawn"birchnutdrake"(坚果鸭)

DebugSpawn"mooseegg"(鹿鸭蛋)