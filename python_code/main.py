import sys
import jieba  # 中文分词工具
from sklearn.feature_extraction.text import TfidfVectorizer  # 创建TF-IDF向量器
from sklearn.metrics.pairwise import cosine_similarity  # 计算余弦相似度
import os

# 读取文件内容
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件错误: {e}")
        sys.exit(1)

# 文本预处理：分词并过滤空白字符
def p_text(text):
    # 使用jieba进行中文分词
    words = jieba.cut(text)
    # 过滤掉空白字符并连接成字符串
    return ' '.join([word for word in words if word.strip()])

# 相似度计算
def cal_sim(orig_text, plag_text):
    # 预处理文本
    orig_p = p_text(orig_text)
    plag_p = p_text(plag_text)

    # 创建TF-IDF向量器
    vec = TfidfVectorizer()
    # 拟合并转换文本
    tfidf_matrix = vec.fit_transform([orig_p, plag_p])

    # 计算余弦相似度
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return sim

# 批量计算相似度，输出到指定答案文件
def check_sim(orig_path, plag_path, answer_path):
    # # 读取原文
    # orig_text = read_file(orig_path)
    # print(f"成功读取原文: {os.path.basename(orig_path)}")
    #
    # # 准备结果内容
    # results = []
    #
    # # 处理每个抄袭版论文
    # for i, plag_path in enumerate(plag_paths):
    #     # 读取抄袭版论文
    #     plag_text = read_file(plag_path)
    #
    #     # 计算相似度
    #     sim = cal_sim(orig_text, plag_text)
    #
    #     # 四舍五入保留两位小数
    #     sim_rate = round(sim, 2)
    #
    #     # 记录结果
    #     filename = os.path.basename(plag_path)
    #     results.append(f"{filename}: {sim_rate:.2f}") # 保证两位小数
    #     print(f"第{i + 1}个抄袭版论文: {filename}的重复率: {sim_rate:.2f}")

    # 读取原文和抄袭文件
    orig_text = read_file(orig_path)
    plag_text = read_file(plag_path)
    print(f"成功读取原文: {os.path.basename(orig_path)}")
    print(f"成功读取抄袭文件: {os.path.basename(plag_path)}")

    # 检查文本是否为空
    if not orig_text.strip():
        print("错误：原文文件内容为空")
        sys.exit(1)
    if not plag_text.strip():
        print("错误：抄袭文件内容为空")
        sys.exit(1)

    # 计算相似度
    sim = cal_sim(orig_text, plag_text)
    sim_rate = round(sim, 2)  # 四舍五入保留两位小数

    # 准备结果内容
    result = f"{os.path.basename(plag_path)}: {sim_rate:.2f}"
    print(f"重复率: {sim_rate:.2f}")

    # 写入结果到答案文件
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(answer_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(answer_path, 'w', encoding='utf-8') as file:
            file.write(result)

        print(f"\n结果已写入答案文件: {os.path.basename(answer_path)}")
    except Exception as e:
        print(f"写入答案文件错误: {e}")
        sys.exit(1)


# def main():
#     # 定义文件路径
#     # 原文绝对路径
#     orig_path = r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig.txt"
#
#     # 抄袭版论文绝对路径列表
#     plag_paths = [
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_0.8_add.txt",
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_0.8_del.txt",
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_0.8_dis_1.txt",
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_0.8_dis_10.txt",
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_0.8_dis_15.txt"
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_copy_full.txt"
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_empty.txt"
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_fragment_mix.txt"
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_plot_only.txt"
#         r"C:\Users\21566\Desktop\rgfirstp\测试文本\orig_word_replace.txt"
#     ]
#
#     # 答案文件绝对路径
#     answer_path = r"C:\Users\21566\Desktop\rgfirstp\answer.txt"
#
#     # 执行批量查重
#     check_sim(orig_path, plag_paths, answer_path)

def main():
    # 检查命令行参数是否足够
    if len(sys.argv) != 4:
        print("使用方式: python main.py [原文文件] [抄袭版论文的文件] [答案文件]")
        sys.exit(1)

    # 解析命令行参数
    orig_path = sys.argv[1]         # 第一个参数：原文路径
    plag_path = sys.argv[2]         # 第二个参数：抄袭文件路径
    answer_path = sys.argv[3]       # 第三个参数：结果输出路径

    # 验证文件是否存在
    if not os.path.isfile(orig_path):
        print(f"错误：原文文件不存在 - {orig_path}")
        sys.exit(1)
    if not os.path.isfile(plag_path):
        print(f"错误：抄袭文件不存在 - {plag_path}")
        sys.exit(1)

    # 执行查重
    check_sim(orig_path, plag_path, answer_path)


if __name__ == "__main__":
    main()
