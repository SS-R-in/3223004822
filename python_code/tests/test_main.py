import unittest
import os
from main import read_file, p_text, cal_sim, check_sim

# -------------- 全局配置：测试文件路径--------------
# 测试文本文件夹路径
TEST_TEXT_DIR = r"C:\Users\21566\Desktop\rgfirstp\测试文本"  # 替换为你的 test_texts 实际路径
# 原文路径（基准文件）
ORIG_PATH = os.path.join(TEST_TEXT_DIR, "orig.txt")
# 所有抄袭文本路径（覆盖不同篡改类型）
PLAG_PATHS = [
    os.path.join(TEST_TEXT_DIR, "orig_0.8_add.txt"),       # 新增内容
    os.path.join(TEST_TEXT_DIR, "orig_0.8_del.txt"),       # 删除内容
    os.path.join(TEST_TEXT_DIR, "orig_0.8_dis_1.txt"),     # 轻微语序打乱
    os.path.join(TEST_TEXT_DIR, "orig_0.8_dis_10.txt"),    # 中度语序打乱
    os.path.join(TEST_TEXT_DIR, "orig_0.8_dis_15.txt"),    # 重度语序打乱
    os.path.join(TEST_TEXT_DIR, "orig_copy_full.txt"),     # 完全复制（相似度≈1.0）
    os.path.join(TEST_TEXT_DIR, "orig_empty.txt"),         # 空文件
    os.path.join(TEST_TEXT_DIR, "orig_fragment_mix.txt"),  # 碎片化拼接
    os.path.join(TEST_TEXT_DIR, "orig_plot_only.txt"),     # 仅保留情节（改写幅度大）
    os.path.join(TEST_TEXT_DIR, "orig_word_replace.txt")   # 词汇替换（结构保留）
]
# 测试结果输出路径（临时文件夹，避免污染项目）
TEST_OUTPUT_DIR = os.path.join(TEST_TEXT_DIR, "test_answers")
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在

# 测试 main.py 所有核心功能的测试类"
class TestMainFunctions(unittest.TestCase):

    # -------------------------- 1. 测试 read_file 函数 --------------------------
    # 测试正常读取存在的文本文件"
    def test_read_file_normal(self):
        # 读取原文文件
        text = read_file(ORIG_PATH)
        # 断言：读取结果非空，且包含原文特征内容（如《活着》前言的标志性句子）
        self.assertIsNotNone(text, "读取文件返回 None，可能文件为空或读取失败")
        self.assertIn("一位真正的作家永远只为内心写作", text, "读取内容不匹配原文，可能文件路径错误")
    # 测试读取不存在的文件（预期触发程序退出，捕获 sys.exit 信号）
    def test_read_file_not_exist(self):
        # 构造不存在的文件路径
        non_exist_path = os.path.join(TEST_TEXT_DIR, "non_exist.txt")
        # 捕获 sys.exit 信号（main.py 中读取失败会调用 sys.exit(1)）
        with self.assertRaises(SystemExit) as cm:
            read_file(non_exist_path)
        # 断言：退出码为 1（表示读取错误）
        self.assertEqual(cm.exception.code, 1, "读取不存在文件时，未按预期退出")
    # 测试读取空文件（预期触发程序退出）
    def test_read_file_empty(self):
        # 创建临时空文件
        empty_path = os.path.join(TEST_OUTPUT_DIR, "empty.txt")
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("")  # 写入空内容
        # 调用 check_sim 时，空文件会触发退出（read_file 本身不校验空，check_sim 会校验）
        answer_path = os.path.join(TEST_OUTPUT_DIR, "empty_answer.txt")
        with self.assertRaises(SystemExit) as cm:
            check_sim(empty_path, PLAG_PATHS[0], answer_path)
        self.assertEqual(cm.exception.code, 1, "读取空文件时，未按预期退出")
        # 清理临时文件
        os.remove(empty_path)

    # -------------------------- 2. 测试 p_text 函数 --------------------------
    # 测试文本预处理（结巴分词+过滤空白字符）
    def test_p_text_jieba_cut(self):
        # 原始文本（含中文、空格、换行）
        raw_text = "活着 前言\n一位真正的作家 永远只为内心写作"
        # 调用预处理函数
        processed_text = p_text(raw_text)
        # 断言：分词后无空白字符，且关键词被正确拆分（如“作家”“内心”单独成词）
        self.assertNotIn("\n", processed_text, "预处理未过滤换行符")
        self.assertNotIn("  ", processed_text, "预处理未过滤多余空格")
        self.assertIn("活着 前言 一位 真正 的 作家", processed_text, "结巴分词未生效")

    # -------------------------- 3. 测试 cal_sim 函数 --------------------------
    # 测试完全复制文本的相似度（预期≈1.0）
    def test_cal_sim_full_copy(self):
        # 读取完全复制的文本（orig_copy_full.txt）
        orig_text = read_file(ORIG_PATH)
        plag_text = read_file(os.path.join(TEST_TEXT_DIR, "orig_copy_full.txt"))
        # 计算相似度
        sim = cal_sim(orig_text, plag_text)
        # 断言：相似度接近 1.0（允许微小误差，因分词/TF-IDF 计算的浮点精度）
        self.assertAlmostEqual(sim, 1.0, delta=0.05, msg="完全复制文本相似度未达预期")
    # 测试词汇替换文本的相似度（预期≈0.8-0.9，因结构保留）
    def test_cal_sim_word_replace(self):
        orig_text = read_file(ORIG_PATH)
        plag_text = read_file(os.path.join(TEST_TEXT_DIR, "orig_word_replace.txt"))
        sim = cal_sim(orig_text, plag_text)
        # 断言：相似度在合理区间（词汇替换未改变核心语义，相似度应较高）
        self.assertGreaterEqual(sim, 0.8, msg="词汇替换文本相似度低于预期")
        self.assertLessEqual(sim, 0.95, msg="词汇替换文本相似度高于预期")
    # 测试仅保留情节的文本相似度（预期≈0.5-0.7，因改写幅度大）
    def test_cal_sim_plot_only(self):
        orig_text = read_file(ORIG_PATH)
        plag_text = read_file(os.path.join(TEST_TEXT_DIR, "orig_plot_only.txt"))
        sim = cal_sim(orig_text, plag_text)
        # 断言：相似度在合理区间（仅保留情节，语义差异大，相似度应较低）
        self.assertGreaterEqual(sim, 0.5, msg="情节改写文本相似度低于预期")
        self.assertLessEqual(sim, 0.7, msg="情节改写文本相似度高于预期")

    # -------------------------- 4. 测试 check_sim 函数 --------------------------
    # 测试 check_sim 能否正确生成结果文件，并验证内容格式
    def test_check_sim_output_file(self):
        # 选择一个抄袭文本（如 orig_0.8_add.txt）
        plag_path = PLAG_PATHS[0]
        answer_path = os.path.join(TEST_OUTPUT_DIR, "test_answer.txt")
        # 执行查重（生成结果文件）
        check_sim(ORIG_PATH, plag_path, answer_path)
        # 断言：结果文件存在
        self.assertTrue(os.path.exists(answer_path), "结果文件未生成")
        # 读取结果文件，验证格式（预期格式："文件名: 0.XX"）
        with open(answer_path, "r", encoding="utf-8") as f:
            result = f.read().strip()
        # 提取文件名（如 orig_0.8_add.txt）
        plag_filename = os.path.basename(plag_path)
        # 断言：结果包含文件名和两位小数的相似度
        self.assertTrue(result.startswith(plag_filename), "结果文件未包含正确的文件名")
        self.assertTrue(result.endswith((".00", ".01", ".02")), "结果格式不是两位小数")  # 覆盖所有两位小数情况
    # 批量测试所有抄袭文本，验证查重流程无异常
    def test_check_sim_batch(self):
        for i, plag_path in enumerate(PLAG_PATHS):
            # 为每个抄袭文本生成独立的结果文件
            answer_path = os.path.join(TEST_OUTPUT_DIR, f"answer_{i}.txt")
            # 执行查重（若有异常会直接报错，中断测试）
            check_sim(ORIG_PATH, plag_path, answer_path)
            # 断言：每个结果文件都存在
            self.assertTrue(os.path.exists(answer_path), f"第{i+1}个抄袭文本的结果文件未生成")


if __name__ == "__main__":
    # 运行所有测试用例（verbosity=2 显示详细测试日志）
    unittest.main(verbosity=2)