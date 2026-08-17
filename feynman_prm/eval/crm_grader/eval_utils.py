import math 
import re 

from .eval_grader import math_equal 
from .eval_normalizer import extract_math_answer_new 


def eval_gsm8k (scored_results ,print_acc =False ,answers =None ,is_extract =False ):
    ans_re =re .compile (r"#### (\-?[0-9\.\,]+)")
    invalid_answer ="[invalid]"

    def extract_answer_hf (completion ):
        match =ans_re .search (completion )
        if match :
            match_str =match .group (1 ).strip ().replace (",","")
            return eval (match_str )
        return invalid_answer 

    def extract_answer (completion ):
        try :
            last_number =re .findall (r"\d+\.\d+|\d+",completion )[-1 ]
            return eval (last_number )
        except Exception :
            return invalid_answer 

    def is_correct (completion ,answer ,is_extract ):
        if is_extract :
            try :
                gold =eval (answer )
            except Exception :
                gold =answer 
        else :
            gold =extract_answer_hf (answer )
        assert gold !=invalid_answer ,f"No ground truth answer found in the document: {answer}"
        return extract_answer (completion )==gold 

    if answers is None :
        raise ValueError ("answers must be provided for GSM8K evaluation")

    completions =[result ["response"]for result in scored_results ]
    acc_list =[is_correct (completion ,answer ,is_extract )for completion ,answer in zip (completions ,answers )]
    acc =100 *sum (acc_list )/len (acc_list )
    if print_acc :
        print ("Accuracy:",acc )
    return acc ,acc_list ,[extract_answer (completion )for completion in completions ]


def eval_math_prm (scored_results ,print_acc =False ,all_problems =None ,is_extract =False ):
    def last_boxed_only_string (string ):
        idx =string .rfind ("\boxed")
        if idx <0 :
            idx =string .rfind ("\fbox")
            if idx <0 :
                return None 

        i =idx 
        left_brace_idx =None 
        right_brace_idx =None 
        num_left_braces_open =0 
        while i <len (string ):
            if string [i ]=="{":
                num_left_braces_open +=1 
                if left_brace_idx is None :
                    left_brace_idx =i 
            elif string [i ]=="}":
                num_left_braces_open -=1 
                if num_left_braces_open ==0 :
                    right_brace_idx =i 
                    break 
            i +=1 

        if left_brace_idx is None or right_brace_idx is None :
            return None 
        return string [left_brace_idx +1 :right_brace_idx ].strip ()

    def match_answer (response ):
        is_matched =False 
        for marker in ["answer:\n","answer:","the answer is: ","the final answer is "]:
            ans_idx =response .lower ().rfind (marker )
            if ans_idx !=-1 :
                is_matched =True 
                response =response [ans_idx +len (marker ):].strip ()
                response =response .replace ("I hope it is correct.","").strip ()
                if response .endswith ("."):
                    response =response [:-1 ]
                if response .endswith ("\n"):
                    response =response [:-2 ]
                break 

        ans_boxed =last_boxed_only_string (response )
        if ans_boxed :
            is_matched =True 
            response =ans_boxed 

        return is_matched ,response 

    if all_problems is None :
        raise ValueError ("all_problems must be provided for MATH evaluation")

    completions =[result ["response"]for result in scored_results ]
    assert len (all_problems )==len (completions ),f"{len(all_problems)}\n{len(completions)}"

    correct =[]
    outputs =[]
    for problem_data ,model_output in zip (all_problems ,completions ):
        try :
            answer =extract_math_answer_new (problem_data ["question"],problem_data ["solution"],is_extract )
            _ ,model_output =match_answer (model_output )
        except Exception :
            model_output =None 
            answer =None 

        outputs .append (model_output )
        try :
            equiv =math_equal (model_output ,answer ,timeout =True )
        except Exception :
            equiv =False 
        correct .append (equiv )

    total =len (all_problems )
    acc =math .fsum (correct )/total *100 
    if print_acc :
        print ("Overall Accuracy = {}/{} = {:.4f}".format (math .fsum (correct ),total ,acc ))
    return acc ,correct ,outputs 
