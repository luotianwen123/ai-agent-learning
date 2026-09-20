from typing import TypedDict,Optional
class Comment(TypedDict):
    id:int
    content:str
    author:str
    reply_to:Optional[int]
def format_comment(c:Comment) -> str:
    if c["reply_to"] is None:
        return f"{c['author']}：{c['content']}"
    else:
        return f"{c['author']}（回复 #{c['reply_to']} ）：{c['content']}"
if __name__ == "__main__":
    comment1 : Comment={"id":101,"content":"写的不错","author":"张三","reply_to":None}
    comment2 : Comment={"id":102,"content":"同意","author":"李四","reply_to":101}
    print(format_comment(comment1))
    print(format_comment(comment2))