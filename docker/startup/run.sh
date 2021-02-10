#!/bin/bash
if [ -n "$GROUP_ID" -a -n "$GROUP_NAME" -a -n "$USER_ID" -a -n "$USER_NAME" ]; then
    addgroup -g $GROUP_ID $GROUP_NAME
    adduser -D -u $USER_ID $USER_NAME -G $GROUP_NAME
fi

 tail -f /dev/null
