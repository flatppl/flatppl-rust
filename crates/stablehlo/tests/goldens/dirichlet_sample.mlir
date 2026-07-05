module {
  func.func @sample(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.constant dense<1.0> : tensor<f32>
    %1 = stablehlo.slice %arg0 [0:1] : (tensor<3xf32>) -> tensor<1xf32>
    %2 = stablehlo.reshape %1 : (tensor<1xf32>) -> tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %2, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %2, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %2 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14 = stablehlo.constant dense<128> : tensor<1xi64>
    %15 = stablehlo.rng %3, %4, %14, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %16 = stablehlo.constant dense<128> : tensor<1xi64>
    %17 = stablehlo.rng %3, %4, %16, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %18 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.constant dense<false> : tensor<i1>
    %20 = stablehlo.constant dense<0.0> : tensor<f32>
    %24:3 = stablehlo.while(%21 = %18, %22 = %19, %23 = %20) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %25 = stablehlo.constant dense<128> : tensor<i32>
      %26 = stablehlo.compare LT, %21, %25, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %27 = stablehlo.not %22 : tensor<i1>
      %28 = stablehlo.and %27, %26 : tensor<i1>
      stablehlo.return %28 : tensor<i1>
    } do {
      %29 = stablehlo.dynamic_slice %15, %21, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %30 = stablehlo.reshape %29 : (tensor<1xf32>) -> tensor<f32>
      %31 = stablehlo.dynamic_slice %17, %21, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
      %33 = stablehlo.multiply %13, %30 : tensor<f32>
      %34 = stablehlo.add %4, %33 : tensor<f32>
      %35 = stablehlo.multiply %34, %34 : tensor<f32>
      %36 = stablehlo.multiply %35, %34 : tensor<f32>
      %37 = stablehlo.multiply %9, %36 : tensor<f32>
      %38 = stablehlo.constant dense<0.5> : tensor<f32>
      %39 = stablehlo.multiply %30, %30 : tensor<f32>
      %40 = stablehlo.multiply %38, %39 : tensor<f32>
      %41 = stablehlo.multiply %9, %36 : tensor<f32>
      %42 = stablehlo.negate %41 : tensor<f32>
      %43 = stablehlo.log %36 : tensor<f32>
      %44 = stablehlo.multiply %9, %43 : tensor<f32>
      %45 = stablehlo.add %40, %9 : tensor<f32>
      %46 = stablehlo.add %45, %42 : tensor<f32>
      %47 = stablehlo.add %46, %44 : tensor<f32>
      %48 = stablehlo.log %32 : tensor<f32>
      %49 = stablehlo.compare LT, %48, %47 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %50 = stablehlo.compare GT, %36, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %51 = stablehlo.and %49, %50 : tensor<i1>
      %52 = stablehlo.constant dense<1> : tensor<i32>
      %53 = stablehlo.add %21, %52 : tensor<i32>
      stablehlo.return %53, %51, %37 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %54 = stablehlo.constant dense<> : tensor<0xi64>
    %55 = stablehlo.rng %3, %4, %54, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %56 = stablehlo.divide %4, %2 : tensor<f32>
    %57 = stablehlo.power %55, %56 : tensor<f32>
    %58 = stablehlo.select %5, %57, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %59 = stablehlo.multiply %24#2, %58 : tensor<f32>
    %60 = stablehlo.divide %59, %0 : tensor<f32>
    %61 = stablehlo.slice %arg0 [1:2] : (tensor<3xf32>) -> tensor<1xf32>
    %62 = stablehlo.reshape %61 : (tensor<1xf32>) -> tensor<f32>
    %63 = stablehlo.constant dense<0.0> : tensor<f32>
    %64 = stablehlo.constant dense<1.0> : tensor<f32>
    %65 = stablehlo.compare LT, %62, %64 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %66 = stablehlo.add %62, %64 : tensor<f32>
    %67 = stablehlo.select %65, %66, %62 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %68 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %69 = stablehlo.subtract %67, %68 : tensor<f32>
    %70 = stablehlo.constant dense<9.0> : tensor<f32>
    %71 = stablehlo.multiply %70, %69 : tensor<f32>
    %72 = stablehlo.sqrt %71 : tensor<f32>
    %73 = stablehlo.divide %64, %72 : tensor<f32>
    %74 = stablehlo.constant dense<128> : tensor<1xi64>
    %75 = stablehlo.rng %63, %64, %74, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %76 = stablehlo.constant dense<128> : tensor<1xi64>
    %77 = stablehlo.rng %63, %64, %76, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %78 = stablehlo.constant dense<0> : tensor<i32>
    %79 = stablehlo.constant dense<false> : tensor<i1>
    %80 = stablehlo.constant dense<0.0> : tensor<f32>
    %84:3 = stablehlo.while(%81 = %78, %82 = %79, %83 = %80) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %85 = stablehlo.constant dense<128> : tensor<i32>
      %86 = stablehlo.compare LT, %81, %85, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %87 = stablehlo.not %82 : tensor<i1>
      %88 = stablehlo.and %87, %86 : tensor<i1>
      stablehlo.return %88 : tensor<i1>
    } do {
      %89 = stablehlo.dynamic_slice %75, %81, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %90 = stablehlo.reshape %89 : (tensor<1xf32>) -> tensor<f32>
      %91 = stablehlo.dynamic_slice %77, %81, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %92 = stablehlo.reshape %91 : (tensor<1xf32>) -> tensor<f32>
      %93 = stablehlo.multiply %73, %90 : tensor<f32>
      %94 = stablehlo.add %64, %93 : tensor<f32>
      %95 = stablehlo.multiply %94, %94 : tensor<f32>
      %96 = stablehlo.multiply %95, %94 : tensor<f32>
      %97 = stablehlo.multiply %69, %96 : tensor<f32>
      %98 = stablehlo.constant dense<0.5> : tensor<f32>
      %99 = stablehlo.multiply %90, %90 : tensor<f32>
      %100 = stablehlo.multiply %98, %99 : tensor<f32>
      %101 = stablehlo.multiply %69, %96 : tensor<f32>
      %102 = stablehlo.negate %101 : tensor<f32>
      %103 = stablehlo.log %96 : tensor<f32>
      %104 = stablehlo.multiply %69, %103 : tensor<f32>
      %105 = stablehlo.add %100, %69 : tensor<f32>
      %106 = stablehlo.add %105, %102 : tensor<f32>
      %107 = stablehlo.add %106, %104 : tensor<f32>
      %108 = stablehlo.log %92 : tensor<f32>
      %109 = stablehlo.compare LT, %108, %107 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %110 = stablehlo.compare GT, %96, %63 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %111 = stablehlo.and %109, %110 : tensor<i1>
      %112 = stablehlo.constant dense<1> : tensor<i32>
      %113 = stablehlo.add %81, %112 : tensor<i32>
      stablehlo.return %113, %111, %97 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %114 = stablehlo.constant dense<> : tensor<0xi64>
    %115 = stablehlo.rng %63, %64, %114, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %116 = stablehlo.divide %64, %62 : tensor<f32>
    %117 = stablehlo.power %115, %116 : tensor<f32>
    %118 = stablehlo.select %65, %117, %64 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %119 = stablehlo.multiply %84#2, %118 : tensor<f32>
    %120 = stablehlo.divide %119, %0 : tensor<f32>
    %121 = stablehlo.slice %arg0 [2:3] : (tensor<3xf32>) -> tensor<1xf32>
    %122 = stablehlo.reshape %121 : (tensor<1xf32>) -> tensor<f32>
    %123 = stablehlo.constant dense<0.0> : tensor<f32>
    %124 = stablehlo.constant dense<1.0> : tensor<f32>
    %125 = stablehlo.compare LT, %122, %124 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %126 = stablehlo.add %122, %124 : tensor<f32>
    %127 = stablehlo.select %125, %126, %122 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %128 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %129 = stablehlo.subtract %127, %128 : tensor<f32>
    %130 = stablehlo.constant dense<9.0> : tensor<f32>
    %131 = stablehlo.multiply %130, %129 : tensor<f32>
    %132 = stablehlo.sqrt %131 : tensor<f32>
    %133 = stablehlo.divide %124, %132 : tensor<f32>
    %134 = stablehlo.constant dense<128> : tensor<1xi64>
    %135 = stablehlo.rng %123, %124, %134, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %136 = stablehlo.constant dense<128> : tensor<1xi64>
    %137 = stablehlo.rng %123, %124, %136, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %138 = stablehlo.constant dense<0> : tensor<i32>
    %139 = stablehlo.constant dense<false> : tensor<i1>
    %140 = stablehlo.constant dense<0.0> : tensor<f32>
    %144:3 = stablehlo.while(%141 = %138, %142 = %139, %143 = %140) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %145 = stablehlo.constant dense<128> : tensor<i32>
      %146 = stablehlo.compare LT, %141, %145, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %147 = stablehlo.not %142 : tensor<i1>
      %148 = stablehlo.and %147, %146 : tensor<i1>
      stablehlo.return %148 : tensor<i1>
    } do {
      %149 = stablehlo.dynamic_slice %135, %141, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %150 = stablehlo.reshape %149 : (tensor<1xf32>) -> tensor<f32>
      %151 = stablehlo.dynamic_slice %137, %141, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %152 = stablehlo.reshape %151 : (tensor<1xf32>) -> tensor<f32>
      %153 = stablehlo.multiply %133, %150 : tensor<f32>
      %154 = stablehlo.add %124, %153 : tensor<f32>
      %155 = stablehlo.multiply %154, %154 : tensor<f32>
      %156 = stablehlo.multiply %155, %154 : tensor<f32>
      %157 = stablehlo.multiply %129, %156 : tensor<f32>
      %158 = stablehlo.constant dense<0.5> : tensor<f32>
      %159 = stablehlo.multiply %150, %150 : tensor<f32>
      %160 = stablehlo.multiply %158, %159 : tensor<f32>
      %161 = stablehlo.multiply %129, %156 : tensor<f32>
      %162 = stablehlo.negate %161 : tensor<f32>
      %163 = stablehlo.log %156 : tensor<f32>
      %164 = stablehlo.multiply %129, %163 : tensor<f32>
      %165 = stablehlo.add %160, %129 : tensor<f32>
      %166 = stablehlo.add %165, %162 : tensor<f32>
      %167 = stablehlo.add %166, %164 : tensor<f32>
      %168 = stablehlo.log %152 : tensor<f32>
      %169 = stablehlo.compare LT, %168, %167 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %170 = stablehlo.compare GT, %156, %123 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %171 = stablehlo.and %169, %170 : tensor<i1>
      %172 = stablehlo.constant dense<1> : tensor<i32>
      %173 = stablehlo.add %141, %172 : tensor<i32>
      stablehlo.return %173, %171, %157 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %174 = stablehlo.constant dense<> : tensor<0xi64>
    %175 = stablehlo.rng %123, %124, %174, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %176 = stablehlo.divide %124, %122 : tensor<f32>
    %177 = stablehlo.power %175, %176 : tensor<f32>
    %178 = stablehlo.select %125, %177, %124 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %179 = stablehlo.multiply %144#2, %178 : tensor<f32>
    %180 = stablehlo.divide %179, %0 : tensor<f32>
    %181 = stablehlo.reshape %60 : (tensor<f32>) -> tensor<1xf32>
    %182 = stablehlo.reshape %120 : (tensor<f32>) -> tensor<1xf32>
    %183 = stablehlo.reshape %180 : (tensor<f32>) -> tensor<1xf32>
    %184 = stablehlo.concatenate %181, %182, %183, dim = 0 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
    %185 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %186 = stablehlo.reduce(%184 init: %185) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %188 = stablehlo.divide %184, %187 : tensor<3xf32>
    return %188 : tensor<3xf32>
  }
}
