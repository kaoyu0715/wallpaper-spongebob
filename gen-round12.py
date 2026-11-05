# 十二代单元出图（issue #25：海绵宝宝腿身比校准）。
# 对症：七代胜出单元实测腿(裤腰→脚底)/身(头顶→裤腰)=1.22（官方经典立绘实测 0.59–0.68，
# 样张海绵宝宝为宽扁块官方早期比例），腿被单元出图环节拉长——prompt 无腿长约束。
# 十二代：prompt 显式加入短腿/长方身措辞 + 负面排长腿；出图后按 §5 硬闸C
# （腿身比 ∈ [0.55,0.75]，sb_proportion.leg_ratio 程序测量）筛种子，再过 §5 单元级必检。
# 用法: python3 gen-round12.py char --role sb-right12 --seeds s1,s2,...
#       python3 gen-round12.py rmbg --role sb-right12 --keep <file>
import importlib.util, os, sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'gen_round11', os.path.join(here, 'gen-round11.py'))
g11 = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['gen-round11.py', 'noop']
spec.loader.exec_module(g11)
sys.argv = _argv
g9 = g11.g9

RAW, CUT = g11.RAW, g11.CUT   # g11raw / g11cut 目录沿用（种子号段区分代际）
# 注册到 gen-round11 的 ROLES（char() 读的是它的模块级 ROLES）
g11.ROLES['sb-right12'] = {'lora': 'sbpilot.safetensors', 'strength': 0.85, 'cfg': 5.5,
  'pos': ('spongebob squarepants, ' + g9.SB_TAGS +
          ', standing on the ground, both feet on the ground, exactly two shoes, '
          'arms relaxed slightly away from body, open hands with four spread fingers, '
          'happy smile, looking to the right, '
          'short thin legs, proportionally small legs, tall rectangular sponge body, '
          'full body visible with margin around, nothing cropped, ' + g9.STYLE),
  'neg_extra': 'three shoes, floating shoe, detached shoe, third shoe, '
               'long legs, long thin legs, big feet'}

char = g11.char
rmbg = g11.rmbg

if __name__ == '__main__':
    args = sys.argv[1:]

    def a(name, default=None):
        return args[args.index('--' + name) + 1] if '--' + name in args else default
    cmd = args[0]
    if cmd == 'char':
        char(a('role'), (a('seeds') or '').split(','))
    elif cmd == 'rmbg':
        rmbg(a('role'), a('keep'))
