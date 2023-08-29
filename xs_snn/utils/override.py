def Override(cls,instance,name,value):
    '''
    未指定参数用 cls全局参数覆盖
    '''
    if value is None:
        instance.__dict__[name]=cls.__dict__[name]
    else:
        instance.__dict__[name]=value