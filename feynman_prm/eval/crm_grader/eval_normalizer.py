import re 


BAD_SUBSTRINGS =["^{","^("]
BAD_REGEXES =["\^[0-9]+\^","\^[0-9][0-9]+"]
TUPLE_CHARS ="()[]"


def _fix_fracs (string ):
    substrs =string .split ("\\frac")
    new_str =substrs [0 ]
    if len (substrs )>1 :
        substrs =substrs [1 :]
        for substr in substrs :
            new_str +="\\frac"
            if len (substr )>0 and substr [0 ]=="{":
                new_str +=substr 
            else :
                try :
                    assert len (substr )>=2 
                except :
                    return string 
                a =substr [0 ]
                b =substr [1 ]
                if b !="{":
                    if len (substr )>2 :
                        post_substr =substr [2 :]
                        new_str +="{"+a +"}{"+b +"}"+post_substr 
                    else :
                        new_str +="{"+a +"}{"+b +"}"
                else :
                    if len (substr )>2 :
                        post_substr =substr [2 :]
                        new_str +="{"+a +"}"+b +post_substr 
                    else :
                        new_str +="{"+a +"}"+b 
    string =new_str 
    return string 


def _str_is_int (x :str )->bool :
    try :
        x =_strip_properly_formatted_commas (x )
        x =float (x )
        return abs (x -int (round (x )))<=1e-7 
    except :
        return False 


def _str_to_int (x :str )->bool :
    x =x .replace (",","")
    if "_"in x :

        x =x .split ("_")[0 ]
    x =float (x )
    return int (x )


def _inject_implicit_mixed_number (step :str ):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 =re .compile ("([0-9]) +([0-9])")
    step =p1 .sub ("\\1+\\2",step )
    return step 


def _strip_properly_formatted_commas (expr :str ):

    p1 =re .compile ("(\d)(,)(\d\d\d)($|\D)")
    while True :
        next_expr =p1 .sub ("\\1\\3\\4",expr )
        if next_expr ==expr :
            break 
        expr =next_expr 
    return next_expr 


def _remove_right_units (expr ):

    if "\\text"in expr :
        try :
            splits =re .split (r"\\text\s*{\s*",expr )

            assert len (splits )==2 and splits [0 ]not in ("","(")
            return splits [0 ]
        except AssertionError :
            pass 

    if "\\text{"in expr :
        return re .sub (r"\\text{([^}]+)}",r"\1",expr )
    elif "\\mbox{"in expr :
        splits =expr .split ("\\mbox{")
        assert len (splits )==2 
        return splits [0 ]
    else :
        return expr 


def _process_and_or_inside_text (string ):
    string =re .sub (r"\s*\\text{\s*(or|and)\s*}\s*",",",string )
    string =re .sub (r",\s*,",",",string )
    return string 


def _remove_left_and_right (expr ):
    """Remove the right and left latex commands."""
    expr =re .sub (r"\\left","",expr )
    expr =re .sub (r"\\right","",expr )
    return expr 


def _fix_sqrt (string ):
    _string =re .sub (r"\\sqrt(\s*\w+)",r"\\sqrt{\1}",string )
    return _string 


def _fix_interval (expr ):
    """Fix interval expression."""
    if "\\in "in expr :
        return expr .split ("\\in ")[1 ].strip ()

    return expr 


def normalize_answer_string (expr :str )->str :
    """Normalize answer expressions."""
    if expr is None :
        return None 


    expr =_remove_left_and_right (expr )
    expr =_process_and_or_inside_text (expr )
    expr =_remove_right_units (expr )
    expr =_fix_interval (expr )
    m =re .search ("^\\\\text\{(?P<text>.+?)\}$",expr )
    if m is not None :
        expr =m .group ("text")

    expr =expr .replace ("\!","")
    expr =expr .replace ("\\%","%")
    expr =expr .replace ("\\$","$")
    expr =expr .replace ("$","")
    expr =expr .replace ("%","")
    expr =expr .replace ("^{\\circ}","")

    expr =expr .replace (" or "," , ")
    expr =expr .replace (" and "," , ")

    expr =expr .replace ("million","*10^6")
    expr =expr .replace ("billion","*10^9")
    expr =expr .replace ("trillion","*10^12")

    for unit in [
    "degree",
    "cm",
    "centimeter",
    "meter",
    "mile",
    "second",
    "minute",
    "hour",
    "week",
    "month",
    "year",
    "foot",
    "feet",
    "inch",
    "yard",
    "p.m.",
    "PM",
    ]:
        expr =re .sub (f"{unit}(es)?(s)? *(\^[0-9]+)?","",expr )

    if "day"in expr :
        days =[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        ]
        weekday_expressed =False 
        for day in days :
            if day in expr :
                weekday_expressed =True 
                break 

        if not weekday_expressed :
            expr =re .sub (f"day(s)?","",expr )

    expr =re .sub (f"\^ *\\\\circ","",expr )

    if len (expr )>0 and expr [0 ]=="{"and expr [-1 ]=="}":
        expr =expr [1 :-1 ]

    expr =_fix_sqrt (expr )
    expr =expr .replace (" ","")


    expr =_fix_fracs (expr )


    expr =re .sub ("- *","-",expr )
    expr =_inject_implicit_mixed_number (expr )

    if _str_is_int (expr ):
        expr =str (_str_to_int (expr ))

    return expr 


def extract_attributes_from_name (file_name ):
    """Extract attributes from file path."""
    eval_set ,problem_type ,fileid =file_name .split ("/")[1 :]
    fileid =fileid .split (".")[0 ]
    return eval_set ,problem_type ,fileid 


def extract_answer_string_2 (answer_str ):
    """For two cases, inside the boxed expression, we needed a second iteration of parsing."""
    left_string ="\\boxed"
    idx =answer_str .rfind (left_string )

    stripped_answer =answer_str [idx +len (left_string ):]
    right_idx =stripped_answer .rfind ("$")

    stripped_answer =stripped_answer [:right_idx ]
    return stripped_answer 


def _post_fix (problem_id ,soln_string ):
    """Post fixing some answer strings"""
    if problem_id =="test/intermediate_algebra/78.json":
        soln_string =re .sub (r"\\(\d+)",r"\1",soln_string )

    return soln_string 

from .eval_grader import extract_answer  # VENDORING FIX: CRM ships `from eval_PQM_grader import ...`, a module absent from its tree
def extract_math_answer_new (question ,response ,direct_answer =False ):
    if direct_answer :
        answer_string =response 
    else :
        answer_string =extract_answer (response )

        if answer_string is None :
            answer_string =extract_answer_string_2 (response )

    parsed_answer =normalize_answer_string (answer_string )
    if not (
    ("Find the equation"in question )
    or ("Enter the equation"in question )
    or ("What is the equation")in question 
    or ("described by the equation")in question 
    or ("Find an equation")in question 
    )and ("="in parsed_answer ):
        if parsed_answer .count ("=")==1 :

            parsed_answer =parsed_answer .split ("=")[1 ]
    return parsed_answer 


if __name__ =="__main__":


    expr_list =["x \\in [-2,7]"]

    for expr in expr_list :
        print (normalize_answer_string (expr ))
