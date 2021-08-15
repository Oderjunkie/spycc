from __future__ import annotations
from ast import parse, Module, FunctionDef, Name, BinOp, arg as astarg, Constant, \
                If, Return, Compare, Call, Expr, AnnAssign, Import, ImportFrom, ClassDef, \
                Assign, Attribute
from ast import Eq, Lt, Gt, LtE, GtE
from ast import Add, Sub, Mult, Div, Mod
from typing import Any, Dict, List, Set, Union
from astunparse import dump as __dump, unparse
from dataclasses import dataclass
from enum import Enum
from os import system
from sys import argv
from filenamelib import temporary_filename
from toml import load as parse_toml

def dump(*args, **kwargs):
    return print(__dump(*args, **kwargs))

globalSettings: Dict[str, Any] = parse_toml('pancake.toml')

class ccast:
    class cctype(Enum):
        void: str = 'void'
        int: str = 'int'
        float: str = 'float'
        bool: str = 'bool'
        char: str = 'char'
        char_p: str = 'char*'
        string: str = 'string'
        auto: str = 'auto'
    class ccop(Enum):
        inc: str = '++'
        dec: str = '--'
        add: str = '+'
        sub: str = '-'
        mul: str = '*'
        div: str = '/'
        mod: str = '%'
        xor: str = '^'
        invert: str = '~'
        lshift: str = '<<'
        rshift: str = '>>'
        binand: str = '&'
        binor: str = '|'
        binnot: str = '!'
        addeq: str = '+='
        subeq: str = '-='
        muleq: str = '*='
        diveq: str = '/='
        modeq: str = '%='
        xoreq: str = '^='
        inverteq: str = '~='
        lshifteq: str = '<<='
        rshifteq: str = '>>='
        binandeq: str = '&='
        binoreq: str = '|='
        binnoteq: str = '!='
        logand: str = '&&'
        logor: str = '||'
        lognot: str = '!'
        deref: str = '*'
        point: str = '&'
        asgn: str = '='
        eq: str = '=='
        ne: str = '!='
        gt: str = '>'
        lt: str = '<'
        gte: str = '>='
        lte: str = '<='
        ns: str = '::'
        member: str = '.'
        member_p: str = '->'
    class name(str):
        def __repr__(self):
            return 'name({0!s})'.format(super().__repr__())
    @dataclass(frozen=True)
    class binop:
        left: Union[ccast.name, str, ccast.binop]
        op: ccast.ccop
        right: Union[ccast.name, str, ccast.binop]
        def __str__(self: ccast.binop) -> str:
            ccop = ccast.ccop
            string = '{0!s}{1!s}{2!s}' if self.op in [ccop.ns, ccop.member, ccop.member_p] else '{0!s} {1!s} {2!s}'
            return string.format(self.left, self.op.value, self.right)
    operand = Union[name, str, binop]
    @dataclass(frozen=True)
    class preop:
        left: ccast.operand
        op: ccast.ccop
        def __str__(self: ccast.binop) -> str:
            return '{0!s}{1!s}'.format(self.left, self.op.value)
    @dataclass(frozen=True)
    class postop:
        op: ccast.ccop
        right: ccast.operand
        def __str__(self: ccast.binop) -> str:
            return '{0!s}{1!s}'.format(self.op.value, self.right)
    @dataclass(frozen=True)
    class ifstat:
        test: ccast.operand
        body: List[ccast.statement]
        orelse: List[ccast.statement]
        def __str__(self: ccast.ifstat):
            if len(self.orelse)==0:
                return 'if ({0!s}) {{\n    {1!s};\n}}'.format(self.test, ';\n'.join([str(inst) for inst in self.body]).replace('\n', '\n    '))
            return 'if ({0!s}) {{\n    {1!s};\n}} else {{\n    {2!s};\n}}'.format(self.test, ';\n'.join([str(inst) for inst in self.body]).replace('\n', '\n    '), '\n'.join([str(inst) for inst in self.orelse]).replace('\n', '\n    '))
    @dataclass(frozen=True)
    class arg:
        argtype: ccast.cctype
        argname: ccast.name
        def __str__(self: ccast.arg) -> str:
            return self.argtype.value + ' ' + self.argname
    @dataclass(frozen=True)
    class functioncall:
        name: ccast.name
        args: List[Union[ccast.operand]]
        def __str__(self: ccast.functioncall) -> str:
            return '{0!s}({1!s})'.format(self.name, ', '.join([str(arg) for arg in self.args]))
    @dataclass(frozen=True)
    class functiondef:
        returntype: ccast.cctype
        name: Union[ccast.name, ccast.binop]
        args: List[ccast.arg]
        body: List[Any]
        islambda: bool = True
        def __str__(self: ccast.functiondef) -> str:
            if self.islambda:
                return 'std::function<{0!s}({1!s})> {2!s} = [&]({3!s}) -> {0!s} {{\n    {4!s};\n}}'.format(self.returntype.value, ', '.join([arg.argtype.value for arg in self.args]), self.name, ', '.join([str(arg) for arg in self.args]), ';\n'.join([str(inst) for inst in self.body]).replace('\n', '\n    '))
            return '{0!s} {1!s}({2!s}) {{\n    {3!s};\n}}'.format(self.returntype.value, self.name, ', '.join([str(arg) for arg in self.args]), ';\n'.join([str(inst) for inst in self.body]).replace('\n', '\n    '))
        def protoize(self: ccast.functiondef) -> str:
            if self.islambda:
                return 'std::function<{0!s}({1!s})> {2!s};'.format(self.returntype.value, ', '.join([arg.argtype.value for arg in self.args]), self.name)
            return '{0!s} {1!s}({2!s});'.format(self.returntype.value, self.name, ', '.join([str(arg) for arg in self.args]))
    @dataclass(frozen=True)
    class constructor:
        name: ccast.name
        args: List[ccast.arg]
        body: List[Any]
        def __str__(self: ccast.functiondef) -> str:
            return '{0!s}({1!s}) {{\n    {2!s};\n}}'.format(self.name, ', '.join([str(arg) for arg in self.args]), ';\n'.join([str(inst) for inst in self.body]).replace('\n', '\n    '))
        def protoize(self: ccast.functiondef) -> str:
            return '{0!s}({1!s});'.format(self.name, ', '.join([str(arg) for arg in self.args]))
    @dataclass(frozen=True)
    class vardefpreset:
        vartype: ccast.cctype
        varname: ccast.name
        vardefault: ccast.operand
        def __str__(self: ccast.vardefpreset) -> str:
            return '{0!s} {1!s} = {2!s}'.format(self.vartype.value, self.varname, self.vardefault)
        def protoize(self: ccast.vardefpreset) -> str:
            return '{0!s} {1!s};'.format(self.vartype.value, self.varname)
    @dataclass(frozen=True)
    class classdef:
        name: ccast.name
        bases: List[ccast.name]
        body: List[ccast.functiondef]
        vars: List[ccast.vardefpreset]
        def __str__(self: ccast.classdef) -> str:
            prefix = lambda x: ccast.binop(self.name, ccast.ccop.ns, x)
            return str('\n'.join([str(ccast.functiondef(funcdef.returntype,
                                                        prefix(funcdef.name),
                                                        funcdef.args,
                                                        funcdef.body,
                                                        funcdef.islambda)) for funcdef in self.body]))
        def protoize(self: ccast.classdef) -> str:
            pycc = globalSettings['py_cc_translation']
            private_prefix = pycc['private_prefix']
            protected_prefix = pycc['protected_prefix']
            public_prefix = pycc['public_prefix']
            privatevardecls = [vardecl for vardecl in self.vars if vardecl.varname.startswith(private_prefix)]
            protectedvardecls = [vardecl for vardecl in self.vars if not vardecl.varname.startswith(private_prefix) and vardecl.varname.startswith(protected_prefix)]
            publicvardecls = [vardecl for vardecl in self.vars if not vardecl.varname.startswith(private_prefix) and not vardecl.varname.startswith(protected_prefix) and vardecl.varname.startswith(public_prefix)]
            privatefuncs = [vardecl for vardecl in self.vars if vardecl.varname.startswith(private_prefix)]
            protectedfuncs = [vardecl for vardecl in self.vars if not vardecl.varname.startswith(private_prefix) and vardecl.varname.startswith(protected_prefix)]
            publicfuncs = [vardecl for vardecl in self.vars if not vardecl.varname.startswith(private_prefix) and not vardecl.varname.startswith(protected_prefix) and vardecl.varname.startswith(public_prefix)]
            compiledstring = ['class {0!s}: {1!s} {{']
            privateamo = len(privatevardecls) + len(privatefuncs)
            protectedamo = len(protectedvardecls) + len(protectedfuncs)
            publicamo = len(publicvardecls) + len(publicfuncs)
            indent = '\n        '
            if privateamo + protectedamo + publicamo > 0:
                compiledstring.append('\n')
            if privateamo > 0:
                compiledstring.append('    private:\n        {2!s}\n')
            if protectedamo > 0:
                compiledstring.append('    protected:\n        {3!s}\n')
            if publicamo > 0:
                compiledstring.append('    public:\n        {4!s}\n')
            compiledstring.append('}}')
            return ''.join(compiledstring).format(
                self.name,
                ', '.join([
                    'public {0!s}'.format(base) for base in self.bases
                ]),
                indent.join([
                    decl.protoize() for decl in [*privatevardecls, *privatefuncs]
                ]),
                indent.join([
                    decl.protoize() for decl in [*protectedvardecls, *protectedfuncs]
                ]),
                indent.join([
                    decl.protoize() for decl in [*publicvardecls, *publicfuncs]
                ])
            )
    @dataclass(frozen=True)
    class vardef:
        vartype: ccast.cctype
        varname: ccast.name
        def __str__(self: ccast.vardef) -> str:
            return '{0!s} {1!s}'.format(self.vartype.value, self.varname)
        def protoize(self: ccast.vardef) -> str:
            return '{0!s} {1!s};'.format(self.vartype.value, self.varname)
    @dataclass(frozen=True)
    class include:
        importname: str
        def __str__(self: ccast.include) -> str:
            return '#include <{0!s}>'.format(self.importname)
        def protoize(self: ccast.include) -> str:
            return '#include <{0!s}>'.format(self.importname)
    @dataclass(frozen=True)
    class includelib:
        importname: str
        def __str__(self: ccast.includelib) -> str:
            return '#include "{0!s}"'.format(self.importname)
        def protoize(self: ccast.includelib) -> str:
            return '' # return '#include "{0!s}"'.format(self.importname)
    @dataclass(frozen=True)
    class module:
        body: List[Union[ccast.functiondef, ccast.include, ccast.includelib]]
        def __str__(self: ccast.module):
            return '\n'.join([
                str(definition) for definition in self.body
            ])
        def protoize(self: ccast.module):
            return '\n'.join([
                definition.protoize() for definition in self.body
            ])
    @dataclass(frozen=True)
    class ret:
        val: Any
        def __str__(self: ccast.ret):
            return 'return {}'.format(self.val)

class cc:
    def __init__(self: cc):
        self.includes: Set[Union[ccast.include, ccast.includelib]] = set()
        self.globalmemory: Dict[str, Dict[str, Union[ccast.cctype, Dict[ccast.name, ccast.cctype]]]] = {}
    def pyt2ct(self: cc, pyt: Union[Name, Constant]) -> ccast.cctype:
        if isinstance(pyt, Name):
            LUT = {
                'int': ccast.cctype.int,
                'float': ccast.cctype.float,
                'str': ccast.cctype.string,
                'bool': ccast.cctype.bool
            }
            return LUT.get(pyt.id, ccast.name(pyt.id))
        if isinstance(pyt, Constant):
            if pyt.value==None:
                return ccast.cctype.void
            raise KeyboardInterrupt('CRAP')
        if pyt == None:
            return ccast.cctype.auto
    def pyc2cc(self: cc, pyc):
        if isinstance(pyc.value, (int, float)):
            return str(pyc.value)
        if isinstance(pyc.value, str):
            return '"' + pyc.value.replace('\n', '\\n') + '"'
        if isinstance(pyc.value, bool):
            return 'true' if pyc.value else 'false'
    def cc(self, src, srcname):
        self.includes = {ccast.includelib(srcname)}
        self.globalmemory = {}
        code = self._cc(parse(src))
        return ccast.module(list(self.includes) + code.body)
    def _cc(self, branch, cascadingmemory:dict={'$scope': '__main__', '$class?': False}):
        if isinstance(branch, Module):
            body = [self._cc(twig, cascadingmemory=cascadingmemory) for twig in branch.body]
            body = [twig for twig in body if twig != None]
            return ccast.module(body)
        if isinstance(branch, FunctionDef):
            returntype = self.pyt2ct(branch.returns)
            name = ccast.name(branch.name)
            outerscope = cascadingmemory.get('$scope', '<unknown>')
            scopename = outerscope+'.'+name
            if scopename == '__main__.main':
                returntype = ccast.cctype.int
            self.globalmemory.setdefault(scopename, {}).setdefault('$returns', returntype)
            args =  [self._cc(twig, cascadingmemory={**cascadingmemory, '$scope': scopename, '$first?': index==0}) for index, twig in enumerate(branch.args.args)]
            insts = [self._cc(twig, cascadingmemory={**cascadingmemory, '$scope': scopename})                      for        twig in           branch.body      ]
            if cascadingmemory.get('$class?', False):
                args = args[1:]
            varsetup = [ccast.vardef(vartype, varname) for varname, vartype in self.globalmemory.get(scopename, {}).get('$localvars', {}).items()]
            self.globalmemory.setdefault(outerscope, {})\
                             .setdefault('$locals', {})\
                             .setdefault(name, ccast.cctype.auto)
            islambda = (outerscope != '__main__') and (not cascadingmemory['$class?'])
            return ccast.functiondef(returntype, name, args, [*varsetup, *insts], islambda = islambda)
        if isinstance(branch, astarg):
            argtype = self.pyt2ct(branch.annotation)
            argname = ccast.name(branch.arg)
            if cascadingmemory.get('$class?', False) and cascadingmemory.get('$first?', False):
                self.globalmemory.setdefault(cascadingmemory.get('$scope', '<unknown>'), {})\
                                 .setdefault('$thisname', argname)
                return None
            self.globalmemory.setdefault(cascadingmemory.get('$scope', '<unknown>'), {})\
                         .setdefault('$args', {})\
                         .setdefault(argname, argtype)
            return ccast.arg(argtype, argname)
        if isinstance(branch, If):
            if isinstance(branch.test, Compare) and\
              len(branch.test.ops) == 1 and\
              isinstance(branch.test.ops[0], Eq) and\
              isinstance(branch.test.left, Name) and\
              branch.test.left.id == '__name__' and\
              len(branch.test.comparators) == 1 and\
              isinstance(branch.test.comparators[0], Constant) and\
              branch.test.comparators[0].value == '__main__':
                return None
            body =   [self._cc(twig, cascadingmemory) for twig in branch.body  ]
            orelse = [self._cc(twig, cascadingmemory) for twig in branch.orelse]
            test = self._cc(branch.test, cascadingmemory)
            return ccast.ifstat(test, body, orelse)
        if isinstance(branch, Return):
            val = self._cc(branch.value, cascadingmemory)
            return ccast.ret(val)
        if isinstance(branch, Constant):
            return self.pyc2cc(branch)
        if isinstance(branch, Name):
            return ccast.name(branch.id)
        if isinstance(branch, Compare):
            left = self._cc(branch.left)
            op = self._cc(branch.ops[0])
            right = self._cc(branch.comparators[0])
            return ccast.binop(left, op, right)
        if isinstance(branch, Call):
            name = self._cc(branch.func)
            args = list(map(self._cc, branch.args))
            if name in ['print', 'getattr', 'setattr', 'exec', 'eval'] and name not in self.globalmemory\
              .setdefault(cascadingmemory.get('$scope', '<unknown>'), {})\
              .setdefault('$localvars', {})\
              .keys():
                if name in ['getattr', 'setattr', 'exec', 'eval']:
                    raise NotImplemented('This is LITERALLY UNCOMPILABLE.\nSTOP.\nObject:\n' + unparse(branch))
                if name == 'print':
                    self.includes.add(ccast.include('iostream'))
                    kws = {kw.arg: kw.value for kw in branch.keywords}
                    kws = {'sep': Constant(value=' '), 'end': Constant(value='\n'), **kws}
                    #print(kws)
                    def printify(arr, original=True):
                        if len(arr) == 0: return ccast.binop(ccast.name('std'), ccast.ccop.ns, ccast.name('cout'))
                        *rest, last = arr
                        start = printify(rest, False) if len(arr) == 1 else ccast.binop(printify(rest, False), ccast.ccop.lshift, self.pyc2cc(kws['sep']))
                        mid = ccast.binop(start, ccast.ccop.lshift, last)
                        final = ccast.binop(mid, ccast.ccop.lshift, self.pyc2cc(kws['end'])) if original else mid
                        return final
                    return printify(args)
            return ccast.functioncall(name, args)
        if isinstance(branch, BinOp):
            left = self._cc(branch.left, cascadingmemory)
            op = self._cc(branch.op, cascadingmemory)
            right = self._cc(branch.right, cascadingmemory)
            return ccast.binop(left, op, right)
        if isinstance(branch, Expr):
            val = self._cc(branch.value, cascadingmemory)
            return val
        if isinstance(branch, AnnAssign):
            target = self._cc(branch.target)
            anno = self.pyt2ct(branch.annotation)
            val = self._cc(branch.value)
            assert self.globalmemory.setdefault(cascadingmemory.get('$scope', '<unknown>'), {})\
                                .setdefault('$localvars', {})\
                                .setdefault(target, anno) == anno,\
                                'code is not type safe lol\nObject:\n' + unparse(branch)
            return ccast.binop(target, ccast.ccop.asgn, val)
        if isinstance(branch, Assign):
            target = self._cc(branch.targets[0])
            val = self._cc(branch.value)
            assert self.globalmemory.setdefault(cascadingmemory.get('$scope', '<unknown>'), {})\
                                .setdefault('$localvars', {})\
                                .setdefault(target, ccast.cctype.auto)
            return ccast.binop(target, ccast.ccop.asgn, val)
        if isinstance(branch, Add):
            return ccast.ccop.add
        if isinstance(branch, Sub):
            return ccast.ccop.sub
        if isinstance(branch, Mult):
            return ccast.ccop.mul
        if isinstance(branch, Div):
            return ccast.ccop.div
        if isinstance(branch, Mod):
            return ccast.ccop.mod
        if isinstance(branch, Eq):
            return ccast.ccop.eq
        if isinstance(branch, Lt):
            return ccast.ccop.lt
        if isinstance(branch, Gt):
            return ccast.ccop.gt
        if isinstance(branch, LtE):
            return ccast.ccop.lte
        if isinstance(branch, GtE):
          return ccast.ccop.gte
        if isinstance(branch, ImportFrom):
            if branch.module == '__future__':
                return None
        if isinstance(branch, Attribute):
            left = self._cc(branch.value, cascadingmemory)
            right = ccast.name(branch.attr)
            metadata = self.globalmemory.get(cascadingmemory.get('$scope', '<unknown>'), {})
            if metadata.get('$thisname', None) == left:
                return ccast.binop(ccast.name('this'), ccast.ccop.member_p, right)
            #else:
                #print(metadata)
                #print(cascadingmemory)
            #print(left == 'self')
            return ccast.binop(left, ccast.ccop.member, right)
        if isinstance(branch, ClassDef):
            name = ccast.name(branch.name)
            new = cascadingmemory.get('$scope', '<unknown>') + '.' + name
            bases = [self._cc(base, {**cascadingmemory, '$scope': new, '$class?': True}) for base in branch.bases]
            bodyanddecls = [self._cc(funcdecl, {**cascadingmemory, '$scope': new, '$class?': True}) for funcdecl in branch.body]
            body = [func for func in bodyanddecls if isinstance(func, ccast.functiondef)]
            annos = self.globalmemory.get(new).get('$localvars', {})
            decls = [ccast.vardefpreset(annos.get(func.left, ccast.cctype.auto), func.left, func.right) for func in bodyanddecls if isinstance(func, ccast.binop) and func.op == ccast.ccop.asgn]
            #print('AA')
            #print(decls)
            return ccast.classdef(name, bases, body, decls)
        dump(branch)
        raise TypeError(type(branch))

def main():
    name = ' '.join(argv[1:])
    with open('{}'.format(name), 'r') as f:
        code = f.read()
        compiler = cc()
        #pprint(compiler.remember)
        #print('--- PYTHON ---', code, '--- C++ ---', compiled, sep='\n\n')
        with temporary_filename('.cc') as filename:
            with temporary_filename('.hh') as header:
                compiled = compiler.cc(code, header)
                #print(compiled)
                #print(compiled.body)
                print(compiled.protoize())
                with open(header, 'w') as f:
                    f.write(compiled.protoize())
                with open(filename, 'w') as f:
                    f.write(str(compiled))
                system('g++ "{}" -o "./{}.exe"'.format(filename, name[::-1].split('.', 1)[-1][::-1]))
        #print('COMMAND: g++ -Ienv env/*.cc -o fibonnaci.exe')

if __name__ == '__main__':
    main()