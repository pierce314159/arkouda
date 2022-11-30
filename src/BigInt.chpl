module BigInt {
    use ServerConfig;

    use Reflection;
    use ServerErrors;
    use Logging;
    use Message;
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use BigInteger;


    private config const logLevel = ServerConfig.logLevel;
    const bLogger = new Logger(logLevel);

    proc bigIntCreationMsg(cmd: string, msgArgs: borrowed MessageArgs, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;

        var size = msgArgs.get("size").getIntValue();
        var len = msgArgs.get("len").getIntValue();
        var arrayNames = msgArgs.get("arrays").getList(size);

        var bigIntArray = makeDistArray(len, bigint);
        // block size is 2**64
        var block_size = 1:bigint;
        block_size <<= 64;
        // for (name, i) in zip(arrayNames, 0..<size by -1) with (+ reduce bigIntArray) {
        for name in arrayNames {
            bigIntArray += toSymEntry(getGenericTypedArrayEntry(name, st), uint).a;
            bigIntArray <<= 64;
        }
        bigIntArray /= block_size;
        var retname = st.nextName();

        // TODO be sure to mod by max_bits

        st.addEntry(retname, new shared SymEntry(bigIntArray));
        var syment = toSymEntry(getGenericTypedArrayEntry(retname, st), bigint);
        writeln(syment.a);
        writeln(syment.dtype);
        repMsg = "created %s".format(st.attrib(retname));
        bLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    use CommandMap;
    registerFunction("big_int_creation",  bigIntCreationMsg, getModuleName());
}