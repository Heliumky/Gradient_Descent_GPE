import re
def extract_real_parts(filename):
    # 读取原始文件
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 处理每一行
    processed_lines = []
    for line in lines:
        # 提取两个复数的实部部分
        matches = re.findall(r'\(([\d\.\+-]+)\+[\d\.\+-]+j\)\s*\(([\d\.\+-]+)\+[\d\.\+-]+j\)', line)
        if not matches:
            # 尝试不同的格式
            matches = re.findall(r'\(([\d\.\+-]+)\s*[\+\-][\d\.\+-]+j\)\s*\(([\d\.\+-]+)\s*[\+\-][\d\.\+-]+j\)', line)
        
        if matches:
            for match in matches:
                real_part1, real_part2 = match
                processed_lines.append(f'{real_part1} {real_part2}\n')
        else:
            # 如果没有匹配到，处理可能是空行或者格式不符合的行
            print(f"No match found for line: {line.strip()}")

    # 保存实数部分到新文件
    with open('output.txt', 'w') as file:
        file.writelines(processed_lines)

# 调用函数
extract_real_parts('GD2_CPUTIME.txt')
