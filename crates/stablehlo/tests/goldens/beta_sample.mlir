module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
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
    %14, %15 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %16 = stablehlo.constant dense<9> : tensor<128xui32>
    %17 = stablehlo.shift_right_logical %15, %16 : tensor<128xui32>
    %18 = stablehlo.convert %17 : (tensor<128xui32>) -> tensor<128xf32>
    %19 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<128xf32>
    %21 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %22 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %23 = stablehlo.multiply %20, %21 : tensor<128xf32>
    %24 = stablehlo.subtract %23, %22 : tensor<128xf32>
    %25 = chlo.erf_inv %24 : tensor<128xf32> -> tensor<128xf32>
    %26 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128xf32>
    %28, %29 = stablehlo.rng_bit_generator %14, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %30 = stablehlo.constant dense<9> : tensor<128xui32>
    %31 = stablehlo.shift_right_logical %29, %30 : tensor<128xui32>
    %32 = stablehlo.convert %31 : (tensor<128xui32>) -> tensor<128xf32>
    %33 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128xf32>
    %35 = stablehlo.constant dense<0> : tensor<i32>
    %36 = stablehlo.constant dense<false> : tensor<i1>
    %37 = stablehlo.constant dense<0.0> : tensor<f32>
    %41:3 = stablehlo.while(%38 = %35, %39 = %36, %40 = %37) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %42 = stablehlo.constant dense<128> : tensor<i32>
      %43 = stablehlo.compare LT, %38, %42, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %44 = stablehlo.not %39 : tensor<i1>
      %45 = stablehlo.and %44, %43 : tensor<i1>
      stablehlo.return %45 : tensor<i1>
    } do {
      %46 = stablehlo.dynamic_slice %27, %38, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %47 = stablehlo.reshape %46 : (tensor<1xf32>) -> tensor<f32>
      %48 = stablehlo.dynamic_slice %34, %38, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %49 = stablehlo.reshape %48 : (tensor<1xf32>) -> tensor<f32>
      %50 = stablehlo.multiply %13, %47 : tensor<f32>
      %51 = stablehlo.add %4, %50 : tensor<f32>
      %52 = stablehlo.multiply %51, %51 : tensor<f32>
      %53 = stablehlo.multiply %52, %51 : tensor<f32>
      %54 = stablehlo.multiply %9, %53 : tensor<f32>
      %55 = stablehlo.constant dense<0.5> : tensor<f32>
      %56 = stablehlo.multiply %47, %47 : tensor<f32>
      %57 = stablehlo.multiply %55, %56 : tensor<f32>
      %58 = stablehlo.multiply %9, %53 : tensor<f32>
      %59 = stablehlo.negate %58 : tensor<f32>
      %60 = stablehlo.log %53 : tensor<f32>
      %61 = stablehlo.multiply %9, %60 : tensor<f32>
      %62 = stablehlo.add %57, %9 : tensor<f32>
      %63 = stablehlo.add %62, %59 : tensor<f32>
      %64 = stablehlo.add %63, %61 : tensor<f32>
      %65 = stablehlo.log %49 : tensor<f32>
      %66 = stablehlo.compare LT, %65, %64 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %67 = stablehlo.compare GT, %53, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %68 = stablehlo.and %66, %67 : tensor<i1>
      %69 = stablehlo.constant dense<1> : tensor<i32>
      %70 = stablehlo.add %38, %69 : tensor<i32>
      stablehlo.return %70, %68, %54 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %71, %72 = stablehlo.rng_bit_generator %28, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %73 = stablehlo.constant dense<9> : tensor<ui32>
    %74 = stablehlo.shift_right_logical %72, %73 : tensor<ui32>
    %75 = stablehlo.convert %74 : (tensor<ui32>) -> tensor<f32>
    %76 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %77 = stablehlo.multiply %75, %76 : tensor<f32>
    %78 = stablehlo.divide %4, %0 : tensor<f32>
    %79 = stablehlo.power %77, %78 : tensor<f32>
    %80 = stablehlo.select %5, %79, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %81 = stablehlo.multiply %41#2, %80 : tensor<f32>
    %82 = stablehlo.divide %81, %2 : tensor<f32>
    %83 = stablehlo.constant dense<0.0> : tensor<f32>
    %84 = stablehlo.constant dense<1.0> : tensor<f32>
    %85 = stablehlo.compare LT, %1, %84 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %86 = stablehlo.add %1, %84 : tensor<f32>
    %87 = stablehlo.select %85, %86, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %88 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %89 = stablehlo.subtract %87, %88 : tensor<f32>
    %90 = stablehlo.constant dense<9.0> : tensor<f32>
    %91 = stablehlo.multiply %90, %89 : tensor<f32>
    %92 = stablehlo.sqrt %91 : tensor<f32>
    %93 = stablehlo.divide %84, %92 : tensor<f32>
    %94, %95 = stablehlo.rng_bit_generator %71, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %96 = stablehlo.constant dense<9> : tensor<128xui32>
    %97 = stablehlo.shift_right_logical %95, %96 : tensor<128xui32>
    %98 = stablehlo.convert %97 : (tensor<128xui32>) -> tensor<128xf32>
    %99 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<128xf32>
    %101 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %102 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %103 = stablehlo.multiply %100, %101 : tensor<128xf32>
    %104 = stablehlo.subtract %103, %102 : tensor<128xf32>
    %105 = chlo.erf_inv %104 : tensor<128xf32> -> tensor<128xf32>
    %106 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %107 = stablehlo.multiply %105, %106 : tensor<128xf32>
    %108, %109 = stablehlo.rng_bit_generator %94, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %110 = stablehlo.constant dense<9> : tensor<128xui32>
    %111 = stablehlo.shift_right_logical %109, %110 : tensor<128xui32>
    %112 = stablehlo.convert %111 : (tensor<128xui32>) -> tensor<128xf32>
    %113 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %114 = stablehlo.multiply %112, %113 : tensor<128xf32>
    %115 = stablehlo.constant dense<0> : tensor<i32>
    %116 = stablehlo.constant dense<false> : tensor<i1>
    %117 = stablehlo.constant dense<0.0> : tensor<f32>
    %121:3 = stablehlo.while(%118 = %115, %119 = %116, %120 = %117) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %122 = stablehlo.constant dense<128> : tensor<i32>
      %123 = stablehlo.compare LT, %118, %122, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %124 = stablehlo.not %119 : tensor<i1>
      %125 = stablehlo.and %124, %123 : tensor<i1>
      stablehlo.return %125 : tensor<i1>
    } do {
      %126 = stablehlo.dynamic_slice %107, %118, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %127 = stablehlo.reshape %126 : (tensor<1xf32>) -> tensor<f32>
      %128 = stablehlo.dynamic_slice %114, %118, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %129 = stablehlo.reshape %128 : (tensor<1xf32>) -> tensor<f32>
      %130 = stablehlo.multiply %93, %127 : tensor<f32>
      %131 = stablehlo.add %84, %130 : tensor<f32>
      %132 = stablehlo.multiply %131, %131 : tensor<f32>
      %133 = stablehlo.multiply %132, %131 : tensor<f32>
      %134 = stablehlo.multiply %89, %133 : tensor<f32>
      %135 = stablehlo.constant dense<0.5> : tensor<f32>
      %136 = stablehlo.multiply %127, %127 : tensor<f32>
      %137 = stablehlo.multiply %135, %136 : tensor<f32>
      %138 = stablehlo.multiply %89, %133 : tensor<f32>
      %139 = stablehlo.negate %138 : tensor<f32>
      %140 = stablehlo.log %133 : tensor<f32>
      %141 = stablehlo.multiply %89, %140 : tensor<f32>
      %142 = stablehlo.add %137, %89 : tensor<f32>
      %143 = stablehlo.add %142, %139 : tensor<f32>
      %144 = stablehlo.add %143, %141 : tensor<f32>
      %145 = stablehlo.log %129 : tensor<f32>
      %146 = stablehlo.compare LT, %145, %144 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %147 = stablehlo.compare GT, %133, %83 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %148 = stablehlo.and %146, %147 : tensor<i1>
      %149 = stablehlo.constant dense<1> : tensor<i32>
      %150 = stablehlo.add %118, %149 : tensor<i32>
      stablehlo.return %150, %148, %134 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %151, %152 = stablehlo.rng_bit_generator %108, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %153 = stablehlo.constant dense<9> : tensor<ui32>
    %154 = stablehlo.shift_right_logical %152, %153 : tensor<ui32>
    %155 = stablehlo.convert %154 : (tensor<ui32>) -> tensor<f32>
    %156 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %157 = stablehlo.multiply %155, %156 : tensor<f32>
    %158 = stablehlo.divide %84, %1 : tensor<f32>
    %159 = stablehlo.power %157, %158 : tensor<f32>
    %160 = stablehlo.select %85, %159, %84 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %161 = stablehlo.multiply %121#2, %160 : tensor<f32>
    %162 = stablehlo.divide %161, %2 : tensor<f32>
    %163 = stablehlo.add %82, %162 : tensor<f32>
    %164 = stablehlo.divide %82, %163 : tensor<f32>
    return %164, %151 : tensor<f32>, tensor<2xui64>
  }
}
