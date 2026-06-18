"""An intentionally BAD endpoint — used to show what a blocking review looks like.

Do NOT ship this. It exists so the AI reviewer has something real to catch:
  * no authorization check (any caller can delete any account),
  * SQL injection (the id is f-string-interpolated into the query),
  * an unguarded destructive DELETE.
The reviewer should return blockers and recommend do_not_merge.
"""


def delete_account(request, user_id):
    # BUG: no permission check — should verify request.user may delete user_id.
    query = f"DELETE FROM accounts WHERE id = {user_id}"  # BUG: SQL injection
    db.execute(query)  # BUG: irreversible delete with no confirmation/soft-delete
    return {"status": "deleted", "user_id": user_id}
