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
        var max_bits = msgArgs.get("max_bits").getIntValue();

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

        if max_bits != -1 {
            var modBy = 1:bigint;
            modBy <<= max_bits;
            bigIntArray %= modBy;
        }

        st.addEntry(retname, new shared SymEntry(bigIntArray, max_bits));
        var syment = toSymEntry(getGenericTypedArrayEntry(retname, st), bigint);
        writeln(syment.a);
        writeln(syment.dtype);
        writeln(syment.max_bits);
        repMsg = "created %s".format(st.attrib(retname));
        bLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }

    // proc breakBigIntIntoArrays(big_arr[?D] bigint) {
    //     var tmp = big_arr;
    //     // take in a bigint sym entry and return list of uint64 symentries
    //     while | reduce (tmp!=0) {
            
    //     }
    // }

    use CommandMap;
    registerFunction("big_int_creation",  bigIntCreationMsg, getModuleName());
}