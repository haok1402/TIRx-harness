//! Shared registry dispatch and call context.

use tvm_tirx::ir::{CallObj, TensorLoadObj};
use tvm_tirx::tvm_ffi::object::ObjectRef;
use tvm_tirx::tvm_ffi::ObjectRefCore;

use super::super::analyze::util::{
    as_buffer, buffer_scope, dtype_of, ffi_error, oref, unsupported, AResult,
};
use crate::analyze::buffers::call_op_name;
use super::abi;
use super::{Emitter, RustValue, Uniformity};
use crate::decode::Decoded;

pub fn call_args(node: &ObjectRef) -> Vec<ObjectRef> {
    node.downcast_ref::<CallObj>()
        .map(|call| call.args.iter().map(oref).collect())
        .unwrap_or_default()
}

/// A rejected row never reaches emission: Decoded::entry returns its reason.
/// An op NumSim does not model but tolerates (support "unmodelled", e.g. WGMMA). Its effect is not
/// simulated; only the register operands it may write are *defined*, with zero placeholders: every
/// local-register argument, passed directly (accumulators, fenced operands) or through
/// `address_of` (an encoded descriptor). Downstream reads then do not report uninitialized
/// registers that only the skipped op would have written. The analysis lists every such executed op
/// in the kernel manifest's ``unmodelled`` field; the checkers report it, and NumSim refuses to
/// produce numbers for such a kernel, so a placeholder is never mistaken for a result.
pub fn unmodelled(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let op_id = call.source_op_id(emitter)?;
    for argument in &call.args {
        let Some(target) = local_register_target(argument)? else {
            continue;
        };
        let dtype = dtype_of(&target)?;
        let rust_type = crate::tables::expr_rust_type(emitter.ctx.schema, &dtype)?;
        let zero = Emitter::zero_literal(&rust_type)?;
        let placeholder = RustValue::new(
            format!("WarpValue::splat({zero})"),
            rust_type,
            Uniformity::Varying,
        );
        emitter.emit_explicit_buffer_store(&target, placeholder, op_id, None, None, None)?;
    }
    Ok(None)
}

/// The register element an unmodelled op's argument names: a load of a `local` buffer, or the
/// `address_of` one.
fn local_register_target(argument: &ObjectRef) -> AResult<Option<ObjectRef>> {
    let mut load = argument.clone();
    if let Some(call) = argument.downcast_ref::<CallObj>() {
        if call_op_name(call)?.as_deref() != Some("tirx.address_of") || call.args.len() != 1 {
            return Ok(None);
        }
        load = oref(call.args.get(0).ok_or_else(|| ffi_error("array index out of bounds"))?);
    }
    let Some(tensor_load) = load.downcast_ref::<TensorLoadObj>() else {
        return Ok(None);
    };
    let Some(buffer) = as_buffer(&oref(tensor_load.source.clone())) else {
        return Ok(None);
    };
    Ok((buffer_scope(&buffer) == "local").then_some(load))
}

pub fn rejected(emitter: &mut Emitter, call: &Decoded) -> AResult<Option<RustValue>> {
    let entry = call.entry(emitter.ctx)?;
    unsupported(format!("{} is unsupported: {}", call.op_name, entry.reason))
}

impl<'a> Emitter<'a> {
    pub fn with_call_expr<T>(
        &mut self,
        node: &ObjectRef,
        emit: impl FnOnce(&mut Self) -> AResult<T>,
    ) -> AResult<T> {
        self.call_expr_stack.push(node.clone());
        let result = emit(self);
        self.call_expr_stack.pop();
        result
    }

    pub fn emit_call(&mut self, node: &ObjectRef) -> AResult<Option<RustValue>> {
        if self.diagnostics.enabled {
            self.diagnostics.visited_calls.add(node.clone());
        }
        match self.emit_call_inner(node) {
            Err(crate::analyze::util::Failure::Unsupported { message, .. })
                if self.diagnostics.enabled =>
            {
                self.record_unsupported(node, message);
                Err(crate::analyze::util::Failure::Recorded)
            }
            Err(error) => Err(crate::analyze::frontend::with_node_span(error, node)),
            result => result,
        }
    }

    fn emit_call_inner(&mut self, node: &ObjectRef) -> AResult<Option<RustValue>> {
        let params = self.params.clone();
        let decoded = Decoded::new(
            self.ctx,
            node,
            Some(&params),
            self.op_ids.get(node).copied(),
        )?;
        let emit = decoded.entry(self.ctx)?.emit;
        emit(self, &decoded)
    }

    pub(super) fn expr_site(&mut self, node: &ObjectRef) -> String {
        let op_id = self.lowered_instruction_site(node);
        abi::site(op_id as u64)
    }

    pub(super) fn pointer(&mut self, expr: &ObjectRef, label: &str) -> AResult<RustValue> {
        let mut value = self.emit_expr(expr)?;
        if value.rust_type != "PhysicalPtr" {
            value = self.emit_raw_shared_pointer(expr, Some(value), "ctx.active_mask()")?;
        }
        if value.rust_type != "PhysicalPtr" {
            return unsupported(format!("{label} must be a physical SMEM pointer"));
        }
        Ok(value)
    }
}
