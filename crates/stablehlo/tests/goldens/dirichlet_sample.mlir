module {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.compare LT, %2, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.add %2, %0 : tensor<f32>
    %6 = stablehlo.select %4, %5, %2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %8 = stablehlo.subtract %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<9.0> : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.sqrt %10 : tensor<f32>
    %12 = stablehlo.divide %0, %11 : tensor<f32>
    %13, %14 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %15 = stablehlo.constant dense<9> : tensor<128xui32>
    %16 = stablehlo.shift_right_logical %14, %15 : tensor<128xui32>
    %17 = stablehlo.convert %16 : (tensor<128xui32>) -> tensor<128xf32>
    %18 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128xf32>
    %20 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %21 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %22 = stablehlo.multiply %19, %20 : tensor<128xf32>
    %23 = stablehlo.subtract %22, %21 : tensor<128xf32>
    %24 = chlo.erf_inv %23 : tensor<128xf32> -> tensor<128xf32>
    %25 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<128xf32>
    %27, %28 = stablehlo.rng_bit_generator %13, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %29 = stablehlo.constant dense<9> : tensor<128xui32>
    %30 = stablehlo.shift_right_logical %28, %29 : tensor<128xui32>
    %31 = stablehlo.convert %30 : (tensor<128xui32>) -> tensor<128xf32>
    %32 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<128xf32>
    %34 = stablehlo.constant dense<0> : tensor<i32>
    %35 = stablehlo.constant dense<false> : tensor<i1>
    %39:3 = stablehlo.while(%36 = %34, %37 = %35, %38 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %40 = stablehlo.constant dense<128> : tensor<i32>
      %41 = stablehlo.compare LT, %36, %40, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %42 = stablehlo.not %37 : tensor<i1>
      %43 = stablehlo.and %42, %41 : tensor<i1>
      stablehlo.return %43 : tensor<i1>
    } do {
      %44 = stablehlo.dynamic_slice %26, %36, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %45 = stablehlo.reshape %44 : (tensor<1xf32>) -> tensor<f32>
      %46 = stablehlo.dynamic_slice %33, %36, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %47 = stablehlo.reshape %46 : (tensor<1xf32>) -> tensor<f32>
      %48 = stablehlo.multiply %12, %45 : tensor<f32>
      %49 = stablehlo.add %0, %48 : tensor<f32>
      %50 = stablehlo.multiply %49, %49 : tensor<f32>
      %51 = stablehlo.multiply %50, %49 : tensor<f32>
      %52 = stablehlo.multiply %8, %51 : tensor<f32>
      %53 = stablehlo.constant dense<0.5> : tensor<f32>
      %54 = stablehlo.multiply %45, %45 : tensor<f32>
      %55 = stablehlo.multiply %53, %54 : tensor<f32>
      %56 = stablehlo.negate %52 : tensor<f32>
      %57 = stablehlo.log %51 : tensor<f32>
      %58 = stablehlo.multiply %8, %57 : tensor<f32>
      %59 = stablehlo.add %55, %8 : tensor<f32>
      %60 = stablehlo.add %59, %56 : tensor<f32>
      %61 = stablehlo.add %60, %58 : tensor<f32>
      %62 = stablehlo.log %47 : tensor<f32>
      %63 = stablehlo.compare LT, %62, %61 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %64 = stablehlo.compare GT, %51, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %65 = stablehlo.and %63, %64 : tensor<i1>
      %66 = stablehlo.constant dense<1> : tensor<i32>
      %67 = stablehlo.add %36, %66 : tensor<i32>
      stablehlo.return %67, %65, %52 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %68, %69 = stablehlo.rng_bit_generator %27, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %70 = stablehlo.constant dense<9> : tensor<ui32>
    %71 = stablehlo.shift_right_logical %69, %70 : tensor<ui32>
    %72 = stablehlo.convert %71 : (tensor<ui32>) -> tensor<f32>
    %73 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %74 = stablehlo.multiply %72, %73 : tensor<f32>
    %75 = stablehlo.divide %0, %2 : tensor<f32>
    %76 = stablehlo.power %74, %75 : tensor<f32>
    %77 = stablehlo.select %4, %76, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %78 = stablehlo.multiply %39#2, %77 : tensor<f32>
    %79 = stablehlo.divide %78, %0 : tensor<f32>
    %80 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %81 = stablehlo.reshape %80 : (tensor<1xf32>) -> tensor<f32>
    %82 = stablehlo.compare LT, %81, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %83 = stablehlo.add %81, %0 : tensor<f32>
    %84 = stablehlo.select %82, %83, %81 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %85 = stablehlo.subtract %84, %7 : tensor<f32>
    %86 = stablehlo.multiply %9, %85 : tensor<f32>
    %87 = stablehlo.sqrt %86 : tensor<f32>
    %88 = stablehlo.divide %0, %87 : tensor<f32>
    %89, %90 = stablehlo.rng_bit_generator %68, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %91 = stablehlo.constant dense<9> : tensor<128xui32>
    %92 = stablehlo.shift_right_logical %90, %91 : tensor<128xui32>
    %93 = stablehlo.convert %92 : (tensor<128xui32>) -> tensor<128xf32>
    %94 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %95 = stablehlo.multiply %93, %94 : tensor<128xf32>
    %96 = stablehlo.multiply %95, %20 : tensor<128xf32>
    %97 = stablehlo.subtract %96, %21 : tensor<128xf32>
    %98 = chlo.erf_inv %97 : tensor<128xf32> -> tensor<128xf32>
    %99 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<128xf32>
    %101, %102 = stablehlo.rng_bit_generator %89, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %103 = stablehlo.constant dense<9> : tensor<128xui32>
    %104 = stablehlo.shift_right_logical %102, %103 : tensor<128xui32>
    %105 = stablehlo.convert %104 : (tensor<128xui32>) -> tensor<128xf32>
    %106 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %107 = stablehlo.multiply %105, %106 : tensor<128xf32>
    %111:3 = stablehlo.while(%108 = %34, %109 = %35, %110 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %112 = stablehlo.constant dense<128> : tensor<i32>
      %113 = stablehlo.compare LT, %108, %112, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %114 = stablehlo.not %109 : tensor<i1>
      %115 = stablehlo.and %114, %113 : tensor<i1>
      stablehlo.return %115 : tensor<i1>
    } do {
      %116 = stablehlo.dynamic_slice %100, %108, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %117 = stablehlo.reshape %116 : (tensor<1xf32>) -> tensor<f32>
      %118 = stablehlo.dynamic_slice %107, %108, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %119 = stablehlo.reshape %118 : (tensor<1xf32>) -> tensor<f32>
      %120 = stablehlo.multiply %88, %117 : tensor<f32>
      %121 = stablehlo.add %0, %120 : tensor<f32>
      %122 = stablehlo.multiply %121, %121 : tensor<f32>
      %123 = stablehlo.multiply %122, %121 : tensor<f32>
      %124 = stablehlo.multiply %85, %123 : tensor<f32>
      %125 = stablehlo.constant dense<0.5> : tensor<f32>
      %126 = stablehlo.multiply %117, %117 : tensor<f32>
      %127 = stablehlo.multiply %125, %126 : tensor<f32>
      %128 = stablehlo.negate %124 : tensor<f32>
      %129 = stablehlo.log %123 : tensor<f32>
      %130 = stablehlo.multiply %85, %129 : tensor<f32>
      %131 = stablehlo.add %127, %85 : tensor<f32>
      %132 = stablehlo.add %131, %128 : tensor<f32>
      %133 = stablehlo.add %132, %130 : tensor<f32>
      %134 = stablehlo.log %119 : tensor<f32>
      %135 = stablehlo.compare LT, %134, %133 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %136 = stablehlo.compare GT, %123, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %137 = stablehlo.and %135, %136 : tensor<i1>
      %138 = stablehlo.constant dense<1> : tensor<i32>
      %139 = stablehlo.add %108, %138 : tensor<i32>
      stablehlo.return %139, %137, %124 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %140, %141 = stablehlo.rng_bit_generator %101, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %142 = stablehlo.constant dense<9> : tensor<ui32>
    %143 = stablehlo.shift_right_logical %141, %142 : tensor<ui32>
    %144 = stablehlo.convert %143 : (tensor<ui32>) -> tensor<f32>
    %145 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %146 = stablehlo.multiply %144, %145 : tensor<f32>
    %147 = stablehlo.divide %0, %81 : tensor<f32>
    %148 = stablehlo.power %146, %147 : tensor<f32>
    %149 = stablehlo.select %82, %148, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %150 = stablehlo.multiply %111#2, %149 : tensor<f32>
    %151 = stablehlo.divide %150, %0 : tensor<f32>
    %152 = stablehlo.slice %arg0 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %153 = stablehlo.reshape %152 : (tensor<1xf32>) -> tensor<f32>
    %154 = stablehlo.compare LT, %153, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %155 = stablehlo.add %153, %0 : tensor<f32>
    %156 = stablehlo.select %154, %155, %153 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %157 = stablehlo.subtract %156, %7 : tensor<f32>
    %158 = stablehlo.multiply %9, %157 : tensor<f32>
    %159 = stablehlo.sqrt %158 : tensor<f32>
    %160 = stablehlo.divide %0, %159 : tensor<f32>
    %161, %162 = stablehlo.rng_bit_generator %140, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %163 = stablehlo.constant dense<9> : tensor<128xui32>
    %164 = stablehlo.shift_right_logical %162, %163 : tensor<128xui32>
    %165 = stablehlo.convert %164 : (tensor<128xui32>) -> tensor<128xf32>
    %166 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %167 = stablehlo.multiply %165, %166 : tensor<128xf32>
    %168 = stablehlo.multiply %167, %20 : tensor<128xf32>
    %169 = stablehlo.subtract %168, %21 : tensor<128xf32>
    %170 = chlo.erf_inv %169 : tensor<128xf32> -> tensor<128xf32>
    %171 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %172 = stablehlo.multiply %170, %171 : tensor<128xf32>
    %173, %174 = stablehlo.rng_bit_generator %161, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %175 = stablehlo.constant dense<9> : tensor<128xui32>
    %176 = stablehlo.shift_right_logical %174, %175 : tensor<128xui32>
    %177 = stablehlo.convert %176 : (tensor<128xui32>) -> tensor<128xf32>
    %178 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %179 = stablehlo.multiply %177, %178 : tensor<128xf32>
    %183:3 = stablehlo.while(%180 = %34, %181 = %35, %182 = %3) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %184 = stablehlo.constant dense<128> : tensor<i32>
      %185 = stablehlo.compare LT, %180, %184, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %186 = stablehlo.not %181 : tensor<i1>
      %187 = stablehlo.and %186, %185 : tensor<i1>
      stablehlo.return %187 : tensor<i1>
    } do {
      %188 = stablehlo.dynamic_slice %172, %180, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %189 = stablehlo.reshape %188 : (tensor<1xf32>) -> tensor<f32>
      %190 = stablehlo.dynamic_slice %179, %180, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %191 = stablehlo.reshape %190 : (tensor<1xf32>) -> tensor<f32>
      %192 = stablehlo.multiply %160, %189 : tensor<f32>
      %193 = stablehlo.add %0, %192 : tensor<f32>
      %194 = stablehlo.multiply %193, %193 : tensor<f32>
      %195 = stablehlo.multiply %194, %193 : tensor<f32>
      %196 = stablehlo.multiply %157, %195 : tensor<f32>
      %197 = stablehlo.constant dense<0.5> : tensor<f32>
      %198 = stablehlo.multiply %189, %189 : tensor<f32>
      %199 = stablehlo.multiply %197, %198 : tensor<f32>
      %200 = stablehlo.negate %196 : tensor<f32>
      %201 = stablehlo.log %195 : tensor<f32>
      %202 = stablehlo.multiply %157, %201 : tensor<f32>
      %203 = stablehlo.add %199, %157 : tensor<f32>
      %204 = stablehlo.add %203, %200 : tensor<f32>
      %205 = stablehlo.add %204, %202 : tensor<f32>
      %206 = stablehlo.log %191 : tensor<f32>
      %207 = stablehlo.compare LT, %206, %205 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %208 = stablehlo.compare GT, %195, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %209 = stablehlo.and %207, %208 : tensor<i1>
      %210 = stablehlo.constant dense<1> : tensor<i32>
      %211 = stablehlo.add %180, %210 : tensor<i32>
      stablehlo.return %211, %209, %196 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %212, %213 = stablehlo.rng_bit_generator %173, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %214 = stablehlo.constant dense<9> : tensor<ui32>
    %215 = stablehlo.shift_right_logical %213, %214 : tensor<ui32>
    %216 = stablehlo.convert %215 : (tensor<ui32>) -> tensor<f32>
    %217 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %218 = stablehlo.multiply %216, %217 : tensor<f32>
    %219 = stablehlo.divide %0, %153 : tensor<f32>
    %220 = stablehlo.power %218, %219 : tensor<f32>
    %221 = stablehlo.select %154, %220, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %222 = stablehlo.multiply %183#2, %221 : tensor<f32>
    %223 = stablehlo.divide %222, %0 : tensor<f32>
    %224 = stablehlo.reshape %79 : (tensor<f32>) -> tensor<1xf32>
    %225 = stablehlo.reshape %151 : (tensor<f32>) -> tensor<1xf32>
    %226 = stablehlo.reshape %223 : (tensor<f32>) -> tensor<1xf32>
    %227 = stablehlo.concatenate %224, %225, %226, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %228 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %229 = stablehlo.reduce(%227 init: %228) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %231 = stablehlo.divide %227, %230 : tensor<3xf32>
    return %231, %212 : tensor<3xf32>, tensor<2xui64>
  }
}
