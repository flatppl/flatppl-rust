module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %0, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %0, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128x4xui32>
    %18 = stablehlo.convert %17 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128x4xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128x4xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128x4xf32>
    %25 = chlo.erf_inv %24 : tensor<128x4xf32> -> tensor<128x4xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x4xf32>
    %28 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %29 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %30 = stablehlo.multiply %27, %28 : tensor<128x4xf32>
    %31 = stablehlo.add %30, %29 : tensor<128x4xf32>
    %32, %33 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %34 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %35 = stablehlo.shift_right_logical %33, %34 : tensor<128x4xui32>
    %36 = stablehlo.convert %35 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %37 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %38 = stablehlo.multiply %36, %37 : tensor<128x4xf32>
    %39 = stablehlo.subtract %4, %3 : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %41 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %42 = stablehlo.multiply %38, %40 : tensor<128x4xf32>
    %43 = stablehlo.add %42, %41 : tensor<128x4xf32>
    %44 = stablehlo.constant dense<0> : tensor<i32>
    %45 = stablehlo.constant dense<false> : tensor<4xi1>
    %46 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %50:3 = stablehlo.while(%47 = %44, %48 = %45, %49 = %46) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %51 = stablehlo.constant dense<128> : tensor<i32>
      %52 = stablehlo.compare LT, %47, %51, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %53 = stablehlo.constant dense<true> : tensor<i1>
      %54 = stablehlo.reduce(%48 init: %53) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %55 = stablehlo.not %54 : tensor<i1>
      %56 = stablehlo.and %52, %55 : tensor<i1>
      stablehlo.return %56 : tensor<i1>
    } do {
      %57 = stablehlo.constant dense<0> : tensor<i32>
      %58 = stablehlo.dynamic_slice %31, %47, %57, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %59 = stablehlo.reshape %58 : (tensor<1x4xf32>) -> tensor<4xf32>
      %60 = stablehlo.constant dense<0> : tensor<i32>
      %61 = stablehlo.dynamic_slice %43, %47, %60, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %62 = stablehlo.reshape %61 : (tensor<1x4xf32>) -> tensor<4xf32>
      %63 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %64 = stablehlo.multiply %63, %59 : tensor<4xf32>
      %65 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %66 = stablehlo.add %65, %64 : tensor<4xf32>
      %67 = stablehlo.multiply %66, %66 : tensor<4xf32>
      %68 = stablehlo.multiply %67, %66 : tensor<4xf32>
      %69 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %70 = stablehlo.multiply %69, %68 : tensor<4xf32>
      %71 = stablehlo.constant dense<0.5> : tensor<f32>
      %72 = stablehlo.multiply %59, %59 : tensor<4xf32>
      %73 = stablehlo.broadcast_in_dim %71, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %74 = stablehlo.multiply %73, %72 : tensor<4xf32>
      %75 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %76 = stablehlo.multiply %75, %68 : tensor<4xf32>
      %77 = stablehlo.negate %76 : tensor<4xf32>
      %78 = stablehlo.log %68 : tensor<4xf32>
      %79 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %80 = stablehlo.multiply %79, %78 : tensor<4xf32>
      %81 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %82 = stablehlo.add %74, %81 : tensor<4xf32>
      %83 = stablehlo.add %82, %77 : tensor<4xf32>
      %84 = stablehlo.add %83, %80 : tensor<4xf32>
      %85 = stablehlo.log %62 : tensor<4xf32>
      %86 = stablehlo.compare LT, %85, %84 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %87 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %88 = stablehlo.compare GT, %68, %87 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %89 = stablehlo.and %86, %88 : tensor<4xi1>
      %90 = stablehlo.select %48, %49, %70 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %91 = stablehlo.or %48, %89 : tensor<4xi1>
      %92 = stablehlo.constant dense<1> : tensor<i32>
      %93 = stablehlo.add %47, %92 : tensor<i32>
      stablehlo.return %93, %91, %90 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %94, %95 = stablehlo.rng_bit_generator %32, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %96 = stablehlo.constant dense<9> : tensor<4xui32>
    %97 = stablehlo.shift_right_logical %95, %96 : tensor<4xui32>
    %98 = stablehlo.convert %97 : (tensor<4xui32>) -> tensor<4xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<4xf32>
    %101 = stablehlo.subtract %4, %3 : tensor<f32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %103 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %104 = stablehlo.multiply %100, %102 : tensor<4xf32>
    %105 = stablehlo.add %104, %103 : tensor<4xf32>
    %106 = stablehlo.divide %4, %0 : tensor<f32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %108 = stablehlo.power %105, %107 : tensor<4xf32>
    %109 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %110 = stablehlo.select %5, %108, %109 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %111 = stablehlo.multiply %50#2, %110 : tensor<4xf32>
    %112 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %113 = stablehlo.divide %111, %112 : tensor<4xf32>
    %114 = stablehlo.constant dense<0.0> : tensor<f32>
    %115 = stablehlo.constant dense<1.0> : tensor<f32>
    %116 = stablehlo.compare LT, %1, %115 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %117 = stablehlo.add %1, %115 : tensor<f32>
    %118 = stablehlo.select %116, %117, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %119 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %120 = stablehlo.subtract %118, %119 : tensor<f32>
    %121 = stablehlo.constant dense<9.0> : tensor<f32>
    %122 = stablehlo.multiply %121, %120 : tensor<f32>
    %123 = stablehlo.sqrt %122 : tensor<f32>
    %124 = stablehlo.divide %115, %123 : tensor<f32>
    %125, %126 = stablehlo.rng_bit_generator %94, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %127 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %128 = stablehlo.shift_right_logical %126, %127 : tensor<128x4xui32>
    %129 = stablehlo.convert %128 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %130 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %131 = stablehlo.multiply %129, %130 : tensor<128x4xf32>
    %132 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %133 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %134 = stablehlo.multiply %131, %132 : tensor<128x4xf32>
    %135 = stablehlo.subtract %134, %133 : tensor<128x4xf32>
    %136 = chlo.erf_inv %135 : tensor<128x4xf32> -> tensor<128x4xf32>
    %137 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %138 = stablehlo.multiply %136, %137 : tensor<128x4xf32>
    %139 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %140 = stablehlo.broadcast_in_dim %114, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %141 = stablehlo.multiply %138, %139 : tensor<128x4xf32>
    %142 = stablehlo.add %141, %140 : tensor<128x4xf32>
    %143, %144 = stablehlo.rng_bit_generator %125, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %145 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %146 = stablehlo.shift_right_logical %144, %145 : tensor<128x4xui32>
    %147 = stablehlo.convert %146 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %148 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %149 = stablehlo.multiply %147, %148 : tensor<128x4xf32>
    %150 = stablehlo.subtract %115, %114 : tensor<f32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %152 = stablehlo.broadcast_in_dim %114, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %153 = stablehlo.multiply %149, %151 : tensor<128x4xf32>
    %154 = stablehlo.add %153, %152 : tensor<128x4xf32>
    %155 = stablehlo.constant dense<0> : tensor<i32>
    %156 = stablehlo.constant dense<false> : tensor<4xi1>
    %157 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %161:3 = stablehlo.while(%158 = %155, %159 = %156, %160 = %157) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %162 = stablehlo.constant dense<128> : tensor<i32>
      %163 = stablehlo.compare LT, %158, %162, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %164 = stablehlo.constant dense<true> : tensor<i1>
      %165 = stablehlo.reduce(%159 init: %164) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %166 = stablehlo.not %165 : tensor<i1>
      %167 = stablehlo.and %163, %166 : tensor<i1>
      stablehlo.return %167 : tensor<i1>
    } do {
      %168 = stablehlo.constant dense<0> : tensor<i32>
      %169 = stablehlo.dynamic_slice %142, %158, %168, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %170 = stablehlo.reshape %169 : (tensor<1x4xf32>) -> tensor<4xf32>
      %171 = stablehlo.constant dense<0> : tensor<i32>
      %172 = stablehlo.dynamic_slice %154, %158, %171, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %173 = stablehlo.reshape %172 : (tensor<1x4xf32>) -> tensor<4xf32>
      %174 = stablehlo.broadcast_in_dim %124, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %175 = stablehlo.multiply %174, %170 : tensor<4xf32>
      %176 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %177 = stablehlo.add %176, %175 : tensor<4xf32>
      %178 = stablehlo.multiply %177, %177 : tensor<4xf32>
      %179 = stablehlo.multiply %178, %177 : tensor<4xf32>
      %180 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %181 = stablehlo.multiply %180, %179 : tensor<4xf32>
      %182 = stablehlo.constant dense<0.5> : tensor<f32>
      %183 = stablehlo.multiply %170, %170 : tensor<4xf32>
      %184 = stablehlo.broadcast_in_dim %182, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %185 = stablehlo.multiply %184, %183 : tensor<4xf32>
      %186 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %187 = stablehlo.multiply %186, %179 : tensor<4xf32>
      %188 = stablehlo.negate %187 : tensor<4xf32>
      %189 = stablehlo.log %179 : tensor<4xf32>
      %190 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %191 = stablehlo.multiply %190, %189 : tensor<4xf32>
      %192 = stablehlo.broadcast_in_dim %120, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %193 = stablehlo.add %185, %192 : tensor<4xf32>
      %194 = stablehlo.add %193, %188 : tensor<4xf32>
      %195 = stablehlo.add %194, %191 : tensor<4xf32>
      %196 = stablehlo.log %173 : tensor<4xf32>
      %197 = stablehlo.compare LT, %196, %195 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %198 = stablehlo.broadcast_in_dim %114, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %199 = stablehlo.compare GT, %179, %198 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %200 = stablehlo.and %197, %199 : tensor<4xi1>
      %201 = stablehlo.select %159, %160, %181 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %202 = stablehlo.or %159, %200 : tensor<4xi1>
      %203 = stablehlo.constant dense<1> : tensor<i32>
      %204 = stablehlo.add %158, %203 : tensor<i32>
      stablehlo.return %204, %202, %201 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %205, %206 = stablehlo.rng_bit_generator %143, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %207 = stablehlo.constant dense<9> : tensor<4xui32>
    %208 = stablehlo.shift_right_logical %206, %207 : tensor<4xui32>
    %209 = stablehlo.convert %208 : (tensor<4xui32>) -> tensor<4xf32>
    %210 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %211 = stablehlo.multiply %209, %210 : tensor<4xf32>
    %212 = stablehlo.subtract %115, %114 : tensor<f32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %214 = stablehlo.broadcast_in_dim %114, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %215 = stablehlo.multiply %211, %213 : tensor<4xf32>
    %216 = stablehlo.add %215, %214 : tensor<4xf32>
    %217 = stablehlo.divide %115, %1 : tensor<f32>
    %218 = stablehlo.broadcast_in_dim %217, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %219 = stablehlo.power %216, %218 : tensor<4xf32>
    %220 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %221 = stablehlo.select %116, %219, %220 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %222 = stablehlo.multiply %161#2, %221 : tensor<4xf32>
    %223 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %224 = stablehlo.divide %222, %223 : tensor<4xf32>
    %225 = stablehlo.add %113, %224 : tensor<4xf32>
    %226 = stablehlo.divide %113, %225 : tensor<4xf32>
    return %226, %205 : tensor<4xf32>, tensor<2xui64>
  }
}
