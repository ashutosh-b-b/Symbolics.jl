struct Gradient <: Operator
    """The variable or expression to differentiate with respect to."""
    x
   function Gradient(x)
        @assert isarraysymbolic(x) "Gradient should be used when the variable is an array symbolic. Perhaps you mean `Differential` operator."
        new(value(x))
    end
end

function (G::Gradient)(x)
    @assert ndims(x) <= 1 "Gradient operator should not be defined on array variables i.e. vector valued functions. Use `Jacobian` instead."
    x = unwrap(x)
    array_term(G, x, size = size(G.x), ndims = ndims(G.x))
end

SymbolicUtils.promote_symtype(::Gradient, T) = T
SymbolicUtils.isbinop(::Gradient) = false

Base.show(io::IO, G::Gradient) = print(io, "∇_$(G.x)")
Base.nameof(::Gradient) = :Gradient

struct Jacobian <: Operator
    """The variable or expression to differentiate with respect to."""
    x
    function Jacobian(x)
        @assert isarraysymbolic(x) "Jacobian should be used when the variable is an array symbolic. Perhaps you mean `Differential` operator."
        new(value(x))
    end
end

function (J::Jacobian)(x)
    @assert ndims(x) > 0 "Jacobian is defined for vector valued functions. Use `Gradient` instead."
    x = unwrap(x)
    array_term(J, x, size = (size(x, 1), size(J.x, 1)), ndims = 2)
end

SymbolicUtils.promote_symtype(::Jacobian, T) = T
SymbolicUtils.isbinop(::Jacobian) = false

Base.show(io::IO, J::Jacobian) = print(io, "Jacobian_$(J.x)")
Base.nameof(::Jacobian) = :Jacobian

struct Hessian <: Operator
    """The variable or expression to differentiate with respect to."""
    x
    function Hessian(x)
        @assert isarraysymbolic(x) "Hessian should be used when the variable is an array symbolic. Perhaps you mean `Differential^2` operator."
        new(value(x))        
    end
end

function (H::Hessian)(x)
    x = unwrap(x)
    array_term(H, x, size = (size(H.x, 1), size(H.x, 1)), ndims = 2)
end


SymbolicUtils.promote_symtype(::Hessian, T) = T
SymbolicUtils.isbinop(::Hessian) = false

Base.show(io::IO, H::Hessian) = print(io, "∇^2_$(H.x)")
Base.nameof(::Hessian) = :Hessian
