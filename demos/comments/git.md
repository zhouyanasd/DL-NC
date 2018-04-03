git reset --soft  //回退版本将覆盖commit的源代码，保留index file和working tree的源代码。用于修改上传信息可用这条
git reset --mixed  //不带任何参数的git reset，默认使用这种方式。回退版本将覆盖commit和index file的源代码，只保留working tree的源代码。
git reset --hard   //回退版本将覆盖commit、index file和working tree的源代码。（危险操作）