"""
BBH Format Guide - 为每个 BBH subset 提供明确的答案格式指导
"""

# BBH subset 分类
MULTIPLE_CHOICE_SUBSETS = {
    'date_understanding', 'disambiguation_qa', 'geometric_shapes', 'hyperbaton',
    'logical_deduction_five_objects', 'logical_deduction_seven_objects', 
    'logical_deduction_three_objects', 'movie_recommendation', 'penguins_in_a_table',
    'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection',
    'snarks', 'temporal_sequences', 'tracking_shuffled_objects_five_objects',
    'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects'
}

YES_NO_SUBSETS = {'causal_judgement', 'navigate', 'sports_understanding', 'web_of_lies'}
TRUE_FALSE_SUBSETS = {'boolean_expressions'}
VALID_INVALID_SUBSETS = {'formal_fallacies'}
NUMERIC_SUBSETS = {'multistep_arithmetic_two', 'object_counting'}
TEXT_SUBSETS = {'dyck_languages', 'word_sorting'}


def get_format_instruction(bbh_subset):
    """
    根据 BBH subset 返回相应的答案格式指导
    
    Args:
        bbh_subset: BBH 任务的子集名称
        
    Returns:
        格式指导字符串
    """
    
    if bbh_subset in MULTIPLE_CHOICE_SUBSETS:
        return (
            "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: (X)' "
            "where X is the letter of your chosen option (A, B, C, D, E, etc.)."
        )
    
    elif bbh_subset in YES_NO_SUBSETS:
        return (
            "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: Yes' "
            "or 'Therefore, the final answer is: No'."
        )
    
    elif bbh_subset in TRUE_FALSE_SUBSETS:
        return (
            "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: True' "
            "or 'Therefore, the final answer is: False'."
        )
    
    elif bbh_subset in VALID_INVALID_SUBSETS:
        return (
            "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: valid' "
            "or 'Therefore, the final answer is: invalid'."
        )
    
    elif bbh_subset in NUMERIC_SUBSETS:
        return (
            "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: [number]' "
            "where [number] is the numerical answer (e.g., 24, 8, etc.)."
        )
    
    elif bbh_subset in TEXT_SUBSETS:
        if bbh_subset == 'word_sorting':
            return (
                "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: [sorted words]' "
                "where [sorted words] is the alphabetically sorted list of words (e.g., 'apple banana cherry')."
            )
        elif bbh_subset == 'dyck_languages':
            return (
                "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: [brackets]' "
                "where [brackets] is the completed bracket sequence."
            )
    
    # 默认格式
    return (
        "\n\nIMPORTANT: You must end your response with 'Therefore, the final answer is: [your answer]'."
    )


def add_format_instruction_to_prompt(prompt, bbh_subset):
    """
    将格式指导添加到 prompt 中
    
    Args:
        prompt: 原始 prompt 字符串
        bbh_subset: BBH 任务的子集名称
        
    Returns:
        添加了格式指导的 prompt
    """
    if bbh_subset is None:
        return prompt
    
    format_instruction = get_format_instruction(bbh_subset)
    return prompt + format_instruction


# 示例用法
if __name__ == "__main__":
    # 测试不同 subset 的格式指导
    test_subsets = [
        'causal_judgement',
        'formal_fallacies', 
        'date_understanding',
        'multistep_arithmetic_two',
        'word_sorting',
        'boolean_expressions'
    ]
    
    for subset in test_subsets:
        print(f"\n{subset}:")
        print(get_format_instruction(subset))

